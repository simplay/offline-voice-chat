"""
Glue for Moonshine 0.1.5 speech output and recognition events.

All patches of Moonshine objects live here. The attributes used below exist
once AgentFlow.load() has run.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any


class SpeechStream:
    """
    Moonshine's own synthesis thread prints errors and still reports success.
    With our own thread, a failed answer is never counted as delivered.
    """

    def __init__(
        self,
        tts: Any,
        on_chunk: Callable[[Any], None],
        on_error: Callable[[BaseException], None],
    ) -> None:
        self._tts = tts
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._input_ready = threading.Event()
        self._input_finished = threading.Event()
        self._stop_event = threading.Event()
        self._done = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._pump, name="LocalSpeechSynthesis", daemon=True
        )
        self._thread.start()

    def push_text(self, text: str) -> None:
        self._tts.push_text(text)
        self._input_ready.set()

    def flush(self) -> None:
        self._tts.flush()

    def end_input(self) -> None:
        self._tts.end_input()
        self._input_finished.set()
        self._input_ready.set()

    def _pump(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self._input_ready.wait(0.005):
                    continue

                chunk = self._tts.next_chunk()
                if self._stop_event.is_set():
                    return

                if chunk is not None:
                    self._on_chunk(chunk)
                elif self._input_finished.is_set() and not self._tts.is_streaming:
                    return
                else:
                    self._stop_event.wait(0.005)
        except Exception as exception:
            self._error = exception
            self._on_error(exception)
        finally:
            self._done.set()

    def wait(self) -> bool:
        self._done.wait()
        if self._error is not None:
            raise self._error

        self._tts.wait()
        return not self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        signal_playback_stop(self._tts)
        self._tts.cancel_stream()
        self._join()

    def _join(self) -> None:
        # stop() runs on the cancellation thread and __exit__ on the
        # conversation worker, never on the pump thread itself.
        self._thread.join(timeout=2.0)

        if self._thread.is_alive():
            raise RuntimeError("speech synthesis did not stop; restart is required")

    def __enter__(self) -> SpeechStream:
        return self

    def __exit__(
        self, exception_type: object, exception: object, traceback: object
    ) -> None:
        self._stop_event.set()
        self._join()


def start_speech(tts: Any, on_error: Callable[[BaseException], None]) -> SpeechStream:
    from moonshine_voice.tts import _import_say_audio_deps, _PlayItem

    np, _ = _import_say_audio_deps()
    tts._ensure_say_workers()

    def enqueue(chunk: Any) -> None:
        if not chunk.samples:
            return

        item = _PlayItem(
            data=np.asarray(chunk.samples, dtype=np.float32),
            sample_rate=int(chunk.sample_rate),
            device=tts._output_device,
        )

        # Moonshine deliberately buffers just one pending playback chunk.
        # Faster synthesis waits here, on its own worker. A finite wait lets
        # cancellation release the producer even if playback has stopped.
        while not tts._say_stop_event.is_set():
            try:
                tts._play_queue.put(item, timeout=0.04)
                return
            except queue.Full:
                continue

    return SpeechStream(tts, enqueue, on_error)


def add_recognition_listener(
    microphone: Any,
    *,
    on_line_started: Callable[[], None],
    on_line_text: Callable[[str], None],
    on_line_completed: Callable[[], None],
) -> None:
    """
    Moonshine only calls these methods on subclasses of its own listener class.
    That class loads with the speech stack, so our subclass is defined here.
    """

    from moonshine_voice.transcriber import TranscriptEventListener

    class RecognitionListener(TranscriptEventListener):
        def on_line_started(self, event: Any) -> None:
            on_line_started()

        def on_line_updated(self, event: Any) -> None:
            on_line_text(event.line.text)

        def on_line_text_changed(self, event: Any) -> None:
            on_line_text(event.line.text)

        def on_line_completed(self, event: Any) -> None:
            on_line_completed()

        def on_error(self, event: Any) -> None:
            on_line_completed()

    microphone.add_listener(RecognitionListener())


def suppress_expected_tts_abort_error(
    tts: Any,
    on_error: Callable[[BaseException], None],
) -> None:
    """
    Stopping playback on purpose makes PortAudio raise an error; that one is
    hidden. Moonshine prints other errors and continues, so they are reported.
    """

    play_one = tts._play_one
    stop_event = tts._say_stop_event

    def play_one_while_cancellable(*args: object, **kwargs: object) -> Any:
        try:
            return play_one(*args, **kwargs)
        except Exception as exception:
            if stop_event.is_set():
                return None

            on_error(exception)
            raise

    tts._play_one = play_one_while_cancellable


class _InterruptibleOutputStream:
    """
    Short writes let playback stop quickly.
    """

    def __init__(
        self,
        stream: Any,
        stop_event: threading.Event,
        sample_rate: int,
        on_activity: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._stop_event = stop_event
        self._block_samples = max(1, int(sample_rate * 0.04))
        self._on_activity = on_activity

    @property
    def latency(self) -> float:
        return self._stream.latency

    def write(self, data: Any) -> None:
        for offset in range(0, len(data), self._block_samples):
            if self._stop_event.is_set():
                return

            self._stream.write(data[offset : offset + self._block_samples])
            self._on_activity()


def make_tts_interruptible(tts: Any, on_activity: Callable[[], None]) -> None:
    """
    Playback stops after at most one short write.
    """

    acquire_output_stream = tts._acquire_output_stream
    stop_event = tts._say_stop_event

    def acquire_interruptible_stream(
        sounddevice: Any,
        device: int | None,
        sample_rate: int,
    ) -> _InterruptibleOutputStream:
        stream = acquire_output_stream(sounddevice, device, sample_rate)
        return _InterruptibleOutputStream(stream, stop_event, sample_rate, on_activity)

    def stop_cooperatively() -> None:
        stop_event.set()
        for pending_queue in (tts._say_queue, tts._play_queue):
            while True:
                try:
                    pending_queue.get_nowait()
                except queue.Empty:
                    break

                pending_queue.task_done()

        # Aborting a PortAudio MME stream while its worker is inside write()
        # is unsafe. The short writes above see the stop event, and the
        # worker then closes its own stream.
        tts._playback_tail_until = 0.0
        # The workers are None until the first playback starts.
        workers = [
            thread
            for thread in (tts._synth_thread, tts._play_thread)
            if thread is not None
        ]

        for thread in workers:
            thread.join(timeout=2.0)

        if any(thread.is_alive() for thread in workers):
            raise RuntimeError("speech workers did not stop; restart is required")

        tts._synth_thread = None
        tts._play_thread = None

    tts._acquire_output_stream = acquire_interruptible_stream
    tts.stop = stop_cooperatively


def signal_playback_stop(tts: Any) -> None:
    """
    Does not block, so recognition callbacks may call it.
    """

    tts._say_stop_event.set()


def cancel_speech(speech: SpeechStream | None, tts: Any) -> None:
    """
    Blocks until the threads have stopped, so never call it from a recognition
    callback. speech is None while the model is still thinking.
    """

    if speech is None:
        return

    # SpeechStream.stop() signals playback, cancels synthesis, and raises if
    # its worker survives.
    speech.stop()
    tts.stop()
    # Consume Moonshine's pending CANCELLED result before the next reply.
    tts.next_chunk()
