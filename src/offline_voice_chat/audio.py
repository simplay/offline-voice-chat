"""Audio for the Realtime backend: resampling, device sample rates, playback."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# OpenAI Realtime sends and receives 24 kHz mono PCM16.
API_SAMPLE_RATE = 24_000
AUDIO_BLOCK_MS = 20

# The server sends audio faster than real time, so most of a long answer is
# buffered before it plays. This limit only caps memory (about 58 MB at
# 48 kHz); normal answers stay far below it.
MAX_PLAYBACK_BUFFER_SECONDS = 300.0

# Enough queue slots for that duration in 10 ms chunks, so the duration limit
# is always reached first.
PLAYBACK_QUEUE_CHUNKS = int(MAX_PLAYBACK_BUFFER_SECONDS * 1000 / 10)
_DEVICE_SAMPLE_RATES = (24_000, 48_000, 44_100, 32_000, 16_000)


class RealtimePlaybackError(RuntimeError):
    """Raised when Realtime audio cannot be played, for example after a device error."""


class StreamingResampler:
    """Resample mono float32 audio that arrives in chunks, without gaps between them."""

    def __init__(self, source_rate: int, target_rate: int, np: Any) -> None:
        self._source_rate = source_rate
        self._target_rate = target_rate
        self._np = np
        self._buffer = np.empty(0, dtype=np.float32)
        self._next_position = 0.0

    def reset(self) -> None:
        self._buffer = self._np.empty(0, dtype=self._np.float32)
        self._next_position = 0.0

    def convert(self, samples: Any) -> Any:
        np = self._np
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        if source.size == 0:
            return source

        if self._source_rate == self._target_rate:
            return source.copy()

        self._buffer = np.concatenate((self._buffer, source))
        step = self._source_rate / self._target_rate

        # interpolation needs a sample on each side, so the last sample waits for the next chunk.
        available = (self._buffer.size - 1) - self._next_position
        count = max(0, int(np.ceil(available / step)))
        if count == 0:
            return np.empty(0, dtype=np.float32)

        positions = self._next_position + step * np.arange(count)
        sample_positions = np.arange(self._buffer.size, dtype=np.float64)
        interpolated = np.interp(positions, sample_positions, self._buffer)
        result = interpolated.astype(np.float32)

        self._next_position = float(positions[-1] + step)
        discard = min(int(self._next_position), self._buffer.size)

        if discard:
            self._buffer = self._buffer[discard:]
            self._next_position -= discard

        return result


def select_device_rate(
    sounddevice: Any,
    *,
    direction: str,
    device: int | str | None,
) -> int:
    if direction == "input":
        check = sounddevice.check_input_settings
    else:
        check = sounddevice.check_output_settings

    last_error: Exception | None = None
    for rate in _DEVICE_SAMPLE_RATES:
        try:
            check(
                device=device,
                samplerate=rate,
                channels=1,
                dtype="float32",
            )
            return rate
        except (ValueError, sounddevice.PortAudioError) as exception:
            last_error = exception

    raise RuntimeError(
        f"audio {direction} device {device!r} rejected all supported sample rates"
    ) from last_error


@dataclass(frozen=True)
class PlaybackPosition:
    item_id: str
    content_index: int
    audio_end_ms: int


@dataclass(frozen=True)
class _PlaybackChunk:
    epoch: int
    item_id: str
    content_index: int
    samples: Any


class RealtimePlayback:
    """
    Play Realtime audio on its own thread.

    Receiving events, such as the user starting to speak, must never wait for
    playback.
    """

    def __init__(
        self,
        sounddevice: Any,
        np: Any,
        *,
        device: int | str | None,
        on_error: Callable[[BaseException], None] | None = None,
        max_queue_chunks: int = PLAYBACK_QUEUE_CHUNKS,
        max_buffer_seconds: float = MAX_PLAYBACK_BUFFER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sounddevice = sounddevice
        self._np = np
        self._device = device
        self._on_error = on_error
        self._clock = clock
        self._sample_rate = select_device_rate(
            sounddevice, direction="output", device=device
        )
        self._max_input_bytes = int(API_SAMPLE_RATE * max_buffer_seconds) * 2
        self._max_buffer_frames = int(self._sample_rate * max_buffer_seconds)
        self._buffered_frames = 0
        self._resampler = StreamingResampler(API_SAMPLE_RATE, self._sample_rate, np)
        self._resampler_key: tuple[str, int] | None = None

        # None asks the playback worker to stop.
        self._queue: queue.Queue[_PlaybackChunk | None] = queue.Queue(
            maxsize=max_queue_chunks
        )
        self._lock = threading.Lock()
        self._epoch = 0
        self._item_id = ""
        self._content_index = 0
        self._completed_key: tuple[str, int] | None = None
        self._played_frames = 0
        self._last_write_at: float | None = None
        self._playing = False
        self._last_progress = clock()
        self._error: BaseException | None = None
        self._stream: Any | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        self._stream = self._sounddevice.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
            blocksize=max(1, self._sample_rate * AUDIO_BLOCK_MS // 1000),
        )
        self._stream.start()
        self._thread = threading.Thread(
            target=self._play_loop,
            name="RealtimeAudioPlayback",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, item_id: str, content_index: int, pcm16: bytes) -> None:
        if len(pcm16) % 2 or len(pcm16) > self._max_input_bytes:
            raise RealtimePlaybackError("invalid or oversized PCM audio chunk")

        pcm_samples = self._np.frombuffer(pcm16, dtype="<i2")
        source = pcm_samples.astype(self._np.float32) / 32768.0

        with self._lock:
            if self._closed:
                return

            if self._error is not None:
                raise RealtimePlaybackError(str(self._error)) from self._error

            if self._worker_stopped():
                raise RealtimePlaybackError("audio playback worker stopped")

            key = (item_id, content_index)
            idle = not self._playing and self._queue.empty()
            starts_new_part = key != (self._item_id, self._content_index)

            if not self._item_id or (idle and starts_new_part):
                self._item_id = item_id
                self._content_index = content_index
                self._played_frames = 0
                self._last_write_at = None

            if key != self._resampler_key:
                self._resampler.reset()
                self._resampler_key = key

            samples = self._resampler.convert(source)
            if not samples.size:
                return

            if idle:
                # Idle time before this chunk is not a playback stall.
                self._last_progress = self._clock()

            if self._buffered_frames + samples.size > self._max_buffer_frames:
                raise RealtimePlaybackError(
                    "audio playback duration exceeded its buffer limit"
                )

            chunk = _PlaybackChunk(self._epoch, item_id, content_index, samples)
            try:
                self._queue.put_nowait(chunk)
            except queue.Full as exception:
                raise RealtimePlaybackError(
                    "audio playback queue fell behind"
                ) from exception

            self._buffered_frames += samples.size

    def finish_item(self, item_id: str, content_index: int) -> None:
        """
        Mark that the server has sent all audio of this content part.

        After it has fully played, clear() returns None.
        """

        with self._lock:
            self._completed_key = (item_id, content_index)

    def health_error(
        self,
        *,
        stall_timeout: float,
        now: float | None = None,
    ) -> BaseException | None:
        """
        Return worker's error, or an error in case the pending audio stopped playing.
        """

        checked_at = self._clock() if now is None else now
        with self._lock:
            if self._error is not None:
                return self._error

            if self._worker_stopped():
                return RealtimePlaybackError("audio playback worker stopped")

            pending = self._playing or not self._queue.empty()
            if pending and checked_at - self._last_progress >= stall_timeout:
                return RealtimePlaybackError("audio playback stopped making progress")

        return None

    def clear(self) -> PlaybackPosition | None:
        with self._lock:
            position = None

            if self._item_id:
                # Audio still in the device buffer has not been heard yet.
                tail_frames = 0

                if self._last_write_at is not None:
                    seconds_since_write = self._clock() - self._last_write_at
                    tail_seconds = max(0.0, self._stream.latency - seconds_since_write)
                    tail_frames = round(tail_seconds * self._sample_rate)

                current_key = (self._item_id, self._content_index)
                part_complete = current_key == self._completed_key
                heard_completely = (
                    part_complete and self._buffered_frames == 0 and tail_frames == 0
                )

                if not heard_completely:
                    heard_frames = max(0, self._played_frames - tail_frames)
                    # Round down: the server rejects an end past its audio.
                    audio_end_ms = heard_frames * 1000 // self._sample_rate
                    position = PlaybackPosition(
                        item_id=self._item_id,
                        content_index=self._content_index,
                        audio_end_ms=audio_end_ms,
                    )

            self._epoch += 1
            self._item_id = ""
            self._content_index = 0
            self._completed_key = None
            self._played_frames = 0
            self._buffered_frames = 0
            self._last_write_at = None
            self._resampler.reset()
            self._resampler_key = None
            self._drain_queue()
            return position

    def close(self) -> None:
        with self._lock:
            self._closed = True

        # Nothing is enqueued after closing, so the emptied queue has room.
        self.clear()
        self._queue.put_nowait(None)

        thread = self._thread

        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RealtimePlaybackError(
                    "audio worker did not stop; restart is required"
                )

        stream = self._stream

        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

        self._stream = None
        self._thread = None

    def _worker_stopped(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

            self._queue.task_done()

    def _play_loop(self) -> None:
        block_samples = max(1, self._sample_rate * AUDIO_BLOCK_MS // 1000)
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    return

                try:
                    with self._lock:
                        self._playing = True

                    for offset in range(0, item.samples.size, block_samples):
                        with self._lock:
                            if item.epoch != self._epoch:
                                break

                        block = item.samples[offset : offset + block_samples]
                        self._stream.write(block.reshape(-1, 1))

                        with self._lock:
                            self._last_progress = self._clock()
                            if item.epoch != self._epoch:
                                break

                            playing_key = (self._item_id, self._content_index)
                            item_key = (item.item_id, item.content_index)
                            if playing_key != item_key:
                                self._item_id = item.item_id
                                self._content_index = item.content_index
                                self._played_frames = 0

                            self._played_frames += int(block.size)
                            self._last_write_at = self._last_progress
                finally:
                    with self._lock:
                        self._playing = False

                        if item.epoch == self._epoch:
                            self._buffered_frames -= item.samples.size

                    self._queue.task_done()

        except Exception as exception:
            error = RealtimePlaybackError(f"audio playback failed: {exception}")
            with self._lock:
                self._error = error
                self._playing = False

            self._drain_queue()

            if self._on_error is not None:
                self._on_error(error)
