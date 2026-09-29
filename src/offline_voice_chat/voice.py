"""The local voice loop: recognition, answers, speech output, and interruption."""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .commands import STOP_PHRASES, TERMINATE_PHRASES, normalize_command
from .metrics import TurnMetrics
from .moonshine import (
    SpeechStream,
    add_recognition_listener,
    cancel_speech,
    make_tts_interruptible,
    signal_playback_stop,
    start_speech,
    suppress_expected_tts_abort_error,
)
from .reliability import ActiveTurnDeadline
from .speech_text import OpeningSpeechBuffer, is_probable_speaker_echo

_TRANSCRIPT_QUEUE_SIZE = 1
_END_OF_TURN_POLL_SECONDS = 0.05
# Speech that starts this soon after an answer started, before any of it is
# audible, finishes the same request. Later speech is a new question.
_CONTINUATION_SECONDS = 2.0
# The spoken help answer contains these phrases, so the speaker-echo check
# would ignore a user who says them while the help answer plays.
_CONTROL_PHRASES = frozenset((*STOP_PHRASES, *TERMINATE_PHRASES))


@dataclass(frozen=True)
class _QueuedTranscript:
    """
    One spoken turn can arrive as several transcripts; each part keeps its
    epoch so the parts stay in order.
    """

    parts: tuple[tuple[int, str], ...]
    submitted_at: float

    @property
    def text(self) -> str:
        return " ".join(text for _, text in self.parts)

    @property
    def epoch(self) -> int:
        return self.parts[-1][0]


def _join_transcripts(
    first: _QueuedTranscript, second: _QueuedTranscript
) -> _QueuedTranscript:
    parts_by_epoch = sorted(first.parts + second.parts, key=lambda part: part[0])
    submitted_at = max(first.submitted_at, second.submitted_at)
    return _QueuedTranscript(tuple(parts_by_epoch), submitted_at)


_STOP_WORKER = _QueuedTranscript(parts=(), submitted_at=0.0)


class VoiceChatApplication:
    def __init__(
        self,
        stream_reply: Callable[[str], Generator[str, None, None]],
        *,
        language: str,
        voice: str | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        models_dir: Path | None = None,
        speech_update_interval: float = 0.25,
        end_of_turn_delay: float = 1.0,
        cancel_generation: Callable[[], None] | None = None,
        finalize_turn: Callable[[bool], None] | None = None,
        recognition_timeout: float = 30.0,
        generation_stall_timeout: float = 45.0,
        generation_progress_interval: float = 5.0,
        playback_stall_timeout: float = 15.0,
        cancellation_timeout: float = 8.0,
        watchdog_poll_interval: float = 0.25,
        report_latency: bool = False,
        agent_factory: Callable[[], Any] | None = None,
        output: TextIO | None = None,
        error_output: TextIO | None = None,
    ) -> None:
        for name, value in (
            ("recognition_timeout", recognition_timeout),
            ("generation_stall_timeout", generation_stall_timeout),
            ("generation_progress_interval", generation_progress_interval),
            ("playback_stall_timeout", playback_stall_timeout),
            ("cancellation_timeout", cancellation_timeout),
            ("watchdog_poll_interval", watchdog_poll_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if end_of_turn_delay < 0:
            raise ValueError("end_of_turn_delay must not be negative")

        self._stream_reply = stream_reply
        self._language = language
        self._voice = voice
        self._input_device = input_device
        self._output_device = output_device
        self._models_dir = models_dir
        self._speech_update_interval = speech_update_interval
        self._end_of_turn_delay = end_of_turn_delay
        self._cancel_generation = cancel_generation
        self._finalize_turn = finalize_turn
        self._recognition_timeout = recognition_timeout
        self._generation_stall_timeout = generation_stall_timeout
        self._generation_progress_interval = generation_progress_interval
        self._playback_stall_timeout = playback_stall_timeout
        self._cancellation_timeout = cancellation_timeout
        self._watchdog_poll_interval = watchdog_poll_interval
        self._report_latency = report_latency
        self._metrics: TurnMetrics | None = None
        self._agent_factory = agent_factory
        self._output = output if output is not None else sys.stdout
        self._error_output = error_output if error_output is not None else sys.stderr
        self._stop_event = threading.Event()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._recognition_deadline = ActiveTurnDeadline()
        self._response_deadline = ActiveTurnDeadline()
        self._agent: Any | None = None
        self._transcripts: queue.Queue[_QueuedTranscript] = queue.Queue(
            maxsize=_TRANSCRIPT_QUEUE_SIZE
        )
        self._transcript_lock = threading.Lock()
        self._transcript_epoch = 0
        self._conversation_thread: threading.Thread | None = None
        self._active_lock = threading.Lock()
        self._speech_input_lock = threading.Lock()
        self._cleanup_done = threading.Event()
        self._cleanup_done.set()
        self._cleanup_thread: threading.Thread | None = None
        self._resources_safe = True
        self._active_cancel: threading.Event | None = None
        self._active_speech: SpeechStream | None = None
        self._active_tts: Any | None = None
        self._active_response_parts: list[str] = []
        self._active_transcript: str | None = None
        self._active_epoch = 0
        self._active_started_at = 0.0
        self._active_audio_started = False
        # Guarded by _transcript_lock. _line_started_at is None while no
        # recognized line is open.
        self._line_started_at: float | None = None
        self._line_ended_at = float("-inf")
        self._pending_echo_text: str | None = None

    @property
    def can_close_resources(self) -> bool:
        """
        False while a worker may still use the native models.
        """

        worker = self._conversation_thread
        return (
            self._resources_safe
            and self._cleanup_done.is_set()
            and (worker is None or not worker.is_alive())
        )

    def build_agent(self) -> Any:
        if self._agent is not None:
            return self._agent

        factory = self._agent_factory

        if factory is None:
            try:
                from moonshine_voice import AgentFlow
            except ImportError as exception:
                raise RuntimeError(
                    "moonshine-voice is not installed; install this project first"
                ) from exception

            factory = AgentFlow

        agent = (
            factory()
            .language(self._language)
            .use_embeddings(False)
            .barge_in(True)
            .beeps(False)
            .on_progress(self._report_progress)
            .on_error(self._report_error)
            .otherwise(self.submit_transcript)
        )

        if self._voice:
            agent.voice(self._voice)

        # AgentFlow creates its microphone in load(); _configure_microphone()
        # selects the input device after that.
        if self._output_device is not None:
            agent.output_device(self._output_device)

        if self._models_dir is not None:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            agent.models_from(self._models_dir)

        self._agent = agent
        return agent

    def submit_transcript(self, text: str) -> None:
        """
        Moonshine calls this for each completed line. It stops the current answer first.
        """

        transcript = text.strip()
        if not transcript or self._stop_event.is_set():
            return

        self._recognition_deadline.clear()

        with self._active_lock:
            possible_echo = self._pending_echo_text
            self._pending_echo_text = None

        normalized = normalize_command(transcript)
        is_stop_command = normalized in STOP_PHRASES
        matches_spoken_answer = possible_echo is not None and is_probable_speaker_echo(
            transcript, possible_echo
        )

        if matches_spoken_answer and normalized not in _CONTROL_PHRASES:
            print(
                "\nIgnored probable speaker echo.",
                file=self._output,
                flush=True,
            )
            return

        if normalized in TERMINATE_PHRASES:
            if not matches_spoken_answer:
                print(
                    f"\nYou: {transcript}\nStopping...", file=self._output, flush=True
                )
                self.request_stop()
                return

            # The answer being spoken contains this phrase, so it may be the
            # speaker's echo. Only stop the answer; a repeat quits because
            # nothing is playing then.
            print(
                f"\nStopped the answer. Say “{TERMINATE_PHRASES[0]}” again to quit.",
                file=self._output,
                flush=True,
            )
            is_stop_command = True

        with self._active_lock:
            self._transcript_epoch += 1
            epoch = self._transcript_epoch

        self._interrupt_current_reply()

        if is_stop_command:
            with self._transcript_lock:
                self._drop_pending_transcript()

            return

        thread = self._conversation_thread

        if thread is None or not thread.is_alive():
            self._start_conversation_worker()

        self._queue_transcript(transcript, epoch=epoch)

    def _queue_transcript(self, transcript: str, *, epoch: int) -> None:
        """
        A transcript that the worker has not started yet is joined with this one.
        """

        item = _QueuedTranscript(((epoch, transcript),), time.monotonic())
        with self._transcript_lock:
            try:
                waiting = self._transcripts.get_nowait()
            except queue.Empty:
                self._transcripts.put_nowait(item)
                return

            self._transcripts.task_done()

            if waiting is _STOP_WORKER:
                self._transcripts.put_nowait(waiting)
                return  # A transcript during shutdown must not keep the worker alive.

            self._transcripts.put_nowait(_join_transcripts(waiting, item))

    def _drop_pending_transcript(self) -> None:
        """
        Call with _transcript_lock held. A queued stop request stays in place.
        """

        try:
            item = self._transcripts.get_nowait()
        except queue.Empty:
            return

        self._transcripts.task_done()

        if item is _STOP_WORKER:
            self._transcripts.put_nowait(item)

    def handle_transcript(
        self,
        text: str,
        *,
        epoch: int | None = None,
        submitted_at: float | None = None,
    ) -> None:
        transcript = text.strip()
        if not transcript or self._stop_event.is_set():
            return

        if not self._await_cleanup():
            return

        cancel = threading.Event()
        response_parts: list[str] = []

        with self._active_lock:
            outdated = epoch is not None and epoch != self._transcript_epoch
            if self._stop_event.is_set() or outdated:
                return

            self._active_cancel = cancel
            self._active_speech = None
            self._active_tts = None
            self._active_response_parts = response_parts
            self._active_transcript = transcript
            self._active_epoch = self._transcript_epoch if epoch is None else epoch
            self._active_started_at = time.monotonic()
            self._active_audio_started = False
            self._metrics = TurnMetrics(submitted_at=submitted_at)
            self._metrics.mark("queue")

        print(f"\nYou: {transcript}", file=self._output, flush=True)
        self._print_phase("thinking")
        self._response_deadline.arm(
            "generation",
            self._generation_stall_timeout,
            progress_interval=self._generation_progress_interval,
        )

        delivered = False
        try:
            agent = self.build_agent()
            self._stream_and_speak(agent, transcript, cancel, response_parts)
            delivered = bool(response_parts) and not cancel.is_set()
        except Exception as exception:
            if not cancel.is_set():
                self._report_error(exception)

            self._interrupt_current_reply()
        finally:
            self._finish_turn(cancel, response_parts, delivered)

    def _finish_turn(
        self,
        cancel: threading.Event,
        response_parts: list[str],
        delivered: bool,
    ) -> None:
        """
        Waits for a running cancellation, so an interrupted turn is never reported
        as delivered.
        """

        while True:
            safe = self._await_cleanup()
            with self._active_lock:
                if safe and not self._cleanup_done.is_set():
                    continue  # An interruption arrived just before finalization.

                interrupted = cancel.is_set()
                metrics, self._metrics = self._metrics, None

                if self._active_cancel is cancel:
                    self._active_cancel = None
                    self._active_speech = None
                    self._active_tts = None
                    self._active_response_parts = []
                    self._active_transcript = None
                    self._active_audio_started = False

                break

        if self._finalize_turn is not None:
            try:
                self._finalize_turn(delivered and not interrupted)
            except Exception as exception:
                self._report_error(exception)

        self._response_deadline.clear()

        if response_parts:
            suffix = " [interrupted]" if interrupted else ""
            print(suffix, file=self._output, flush=True)
        elif interrupted:
            self._print_phase("turn interrupted")

        if metrics is not None and self._report_latency:
            metrics.mark("turn finished")
            print(f"[Latency: {metrics.summary()}]", file=self._output, flush=True)

        if not self._stop_event.is_set():
            self._print_phase("listening")

    def _stream_and_speak(
        self,
        agent: Any,
        transcript: str,
        cancel: threading.Event,
        response_parts: list[str],
    ) -> None:
        # Register the reply before an interruption can signal its cancellation.
        # Creating the generator must not run inference or other blocking work.
        with self._active_lock:
            if cancel.is_set():
                return

            reply = self._stream_reply(transcript)

        try:
            self._speak_with_native_tts(agent._tts, reply, cancel, response_parts)
        finally:
            reply.close()

    def _speak_with_native_tts(
        self,
        tts: Any,
        reply: Generator[str, None, None],
        cancel: threading.Event,
        response_parts: list[str],
    ) -> None:
        with self._speech_input_lock:
            if cancel.is_set():
                return

            speech = start_speech(tts, self._on_tts_error)
            with self._active_lock:
                self._active_speech = speech
                self._active_tts = tts

            # Cancellation may have arrived during stream creation.
            if cancel.is_set():
                signal_playback_stop(tts)

        with speech:
            try:
                self._forward_reply(reply, speech, cancel, response_parts)
                self._response_deadline.clear("generation")

                with self._speech_input_lock:
                    should_finish = not cancel.is_set()

                    if should_finish:
                        speech.end_input()

                if should_finish:
                    with self._active_lock:
                        if not cancel.is_set():
                            self._response_deadline.arm(
                                "playback", self._playback_stall_timeout
                            )

                    self._wait_for_speech(speech, cancel)
                    self._response_deadline.clear("playback")
            except Exception as exception:
                # Cancel before leaving the with block: SpeechStream's exit
                # joins a pump that may wait for playback room until stopped.
                if not cancel.is_set():
                    self._report_error(exception)

                self._interrupt_current_reply()
                raise
            finally:
                self._await_cleanup()

    def _forward_reply(
        self,
        reply: Generator[str, None, None],
        speech: SpeechStream,
        cancel: threading.Event,
        response_parts: list[str],
    ) -> None:
        opening = OpeningSpeechBuffer()
        answer_started = False

        for text in reply:
            if not text:
                continue

            with self._active_lock:
                if not cancel.is_set() and self._metrics is not None:
                    self._metrics.mark("first text")

            with self._speech_input_lock:
                if cancel.is_set():
                    return

                # Decide before pushing, so the release point lies between the
                # buffered text and this piece, never inside a word or number.
                if opening.should_flush(text):
                    speech.flush()

                speech.push_text(text)
                response_parts.append(text)

                with self._active_lock:
                    if cancel.is_set():
                        return

                    if answer_started:
                        self._response_deadline.touch("generation")
                    else:
                        self._response_deadline.arm(
                            "generation", self._generation_stall_timeout
                        )

            if not answer_started:
                self._print_phase("answer streaming to speech")
                print("Assistant: ", end="", file=self._output, flush=True)
                answer_started = True

            print(text, end="", file=self._output, flush=True)

    def _wait_for_speech(
        self,
        speech: SpeechStream,
        cancel: threading.Event,
    ) -> None:
        """
        Does not wait forever if a stopped playback hangs.
        """

        done = threading.Event()
        errors: list[Exception] = []

        def wait_for_dependency() -> None:
            try:
                completed = speech.wait()
                if not completed and not cancel.is_set():
                    raise RuntimeError("speech playback did not complete")
            except Exception as exception:
                errors.append(exception)
            finally:
                done.set()

        waiter = threading.Thread(
            target=wait_for_dependency,
            name="LocalSpeechDrain",
            daemon=True,
        )
        waiter.start()

        while not done.wait(self._watchdog_poll_interval):
            if not cancel.is_set():
                continue

            if not done.wait(timeout=2.0):
                self._report_error(
                    "speech playback did not stop safely; stopping the application"
                )
                self._stop_event.set()
                self._resources_safe = False

            return

        if errors:
            raise errors[0]

    def _configure_microphone(self, agent: Any) -> None:
        microphone = agent._mic

        if self._input_device is not None:
            microphone.device(self._input_device)

        microphone.update_interval(self._speech_update_interval)
        # Moonshine creates the stream during load(), so update the already
        # constructed stream as well as the public MicTranscriber setting.
        microphone.mic_stream._update_interval = self._speech_update_interval

    def _interrupt_current_reply(
        self, *, remember_echo: bool = False, keep_question: bool = False
    ) -> None:
        with self._active_lock:
            cancel = self._active_cancel
            if cancel is None:
                return

            if remember_echo and self._active_response_parts:
                self._pending_echo_text = "".join(self._active_response_parts)

            if cancel.is_set():
                return

            cancel.set()
            tts = self._active_tts
            self._cleanup_done.clear()
            self._response_deadline.arm("cancelling", self._cancellation_timeout)
            # The user kept talking before any of the answer was audible, so
            # the question is still part of the current turn.
            answer_age = time.monotonic() - self._active_started_at
            answer_unheard = not self._active_audio_started
            within_continuation = answer_age <= _CONTINUATION_SECONDS
            kept_question = None

            if keep_question and answer_unheard and within_continuation:
                kept_question = self._active_transcript

            kept_epoch = self._active_epoch

        if kept_question:
            self._queue_transcript(kept_question, epoch=kept_epoch)

        # tts is None while the model is still thinking.
        if tts is not None:
            signal_playback_stop(tts)

        if self._cancel_generation is not None:
            try:
                self._cancel_generation()
            except Exception as exception:
                self._report_error(exception)

        def cleanup() -> None:
            try:
                # Wait until no text is being pushed to the synthesizer, then stop.
                # This runs here, so the recognition callback never waits.
                with self._speech_input_lock:
                    with self._active_lock:
                        speech = self._active_speech
                        synthesizer = self._active_tts

                    cancel_speech(speech, synthesizer)
            except Exception as exception:
                self._report_error(exception)
                self._resources_safe = False
                self._stop_event.set()
            finally:
                self._cleanup_done.set()

        self._cleanup_thread = threading.Thread(
            target=cleanup,
            name="LocalSpeechCancellation",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _await_cleanup(self) -> bool:
        if not self._cleanup_done.wait(self._cancellation_timeout):
            self._report_error(
                "speech cancellation did not finish; restart is required"
            )
            self._resources_safe = False
            self._stop_event.set()
            return False

        return self._resources_safe

    def _enable_live_barge_in(self, agent: Any) -> None:
        """
        Lets the user interrupt as soon as their first words are recognized.
        """

        make_tts_interruptible(agent._tts, self._on_tts_activity)
        suppress_expected_tts_abort_error(agent._tts, self._on_tts_error)
        add_recognition_listener(
            agent._mic,
            on_line_started=self._on_line_started,
            on_line_text=self._on_line_text,
            on_line_completed=self._on_line_completed,
        )

    def _on_line_completed(self) -> None:
        self._recognition_deadline.clear("recognition")

        with self._transcript_lock:
            self._line_started_at = None
            self._line_ended_at = time.monotonic()

    def _on_line_started(self) -> None:
        with self._transcript_lock:
            self._line_started_at = time.monotonic()

        # A raw audio onset may be speaker bleed while output is audible, or
        # a cough while the model is thinking. Waiting only for the first
        # partial transcript keeps barge-in prompt, rejects obvious echo, and
        # keeps the question when the line produces no words.
        with self._active_lock:
            active = self._active_cancel
            spoken_text = "".join(self._active_response_parts)

            # Compare each line only with the answer audible when it began, so
            # a line that ends without words leaves no stale reference behind.
            self._pending_echo_text = None

            if active is not None and spoken_text:
                self._pending_echo_text = spoken_text

        self._print_phase("speech detected; recognizing")
        progress_interval = min(5.0, self._recognition_timeout / 2)
        self._recognition_deadline.arm(
            "recognition",
            self._recognition_timeout,
            progress_interval=progress_interval,
        )

    def _on_line_text(self, partial: str) -> None:
        if not partial.strip():
            return

        with self._active_lock:
            active = self._active_cancel
            spoken_text = "".join(self._active_response_parts)

            if active is not None and spoken_text:
                self._pending_echo_text = spoken_text

        if active is None or active.is_set():
            return

        is_control_phrase = normalize_command(partial) in _CONTROL_PHRASES
        is_echo = is_probable_speaker_echo(partial, spoken_text)

        if is_control_phrase or not is_echo:
            self._interrupt_current_reply(remember_echo=True, keep_question=True)

    def _on_tts_activity(self) -> None:
        with self._active_lock:
            if self._metrics is not None:
                self._metrics.mark("first audio write")

            if self._active_cancel is not None:
                self._active_audio_started = True

        self._response_deadline.touch("playback")

    def _on_tts_error(self, error: BaseException) -> None:
        if self._stop_event.is_set():
            return

        detail = str(error).strip() or type(error).__name__
        self._report_error(f"speech playback failed: {detail}")
        self._interrupt_current_reply()

    def _start_conversation_worker(self) -> None:
        thread = self._conversation_thread
        if thread is not None and thread.is_alive():
            return

        self._conversation_thread = threading.Thread(
            target=self._conversation_loop,
            name="VoiceChatConversation",
            daemon=True,
        )
        self._conversation_thread.start()
        self._start_watchdog()

    def _start_watchdog(self) -> None:
        thread = self._watchdog_thread
        if thread is not None and thread.is_alive():
            return

        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="LocalVoiceWatchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self._watchdog_poll_interval):
            self._check_active_turn_health()

    def _check_active_turn_health(self, now: float | None = None) -> None:
        for deadline in (self._recognition_deadline, self._response_deadline):
            signal = deadline.poll(now)
            if signal is None:
                continue

            if signal.kind == "progress":
                if signal.phase == "recognition":
                    self._print_phase("still recognizing; pause briefly to finish")
                elif signal.phase == "generation":
                    self._print_phase("still thinking")

                continue

            if signal.phase == "recognition":
                self._report_error(
                    "speech recognition did not finalize this turn; pause, then "
                    "try speaking again"
                )
                self._print_phase(
                    "recognition stalled; waiting for microphone recovery"
                )
            elif signal.phase == "generation":
                self._report_error(
                    "the local model stopped producing tokens; cancelling the turn"
                )
                self._interrupt_current_reply()
            elif signal.phase == "playback":
                self._report_error(
                    "speech playback stopped making progress; cancelling the turn"
                )
                self._interrupt_current_reply()
            elif signal.phase == "cancelling":
                self._report_error(
                    "the local model did not stop cooperatively; restart is required"
                )
                self._stop_event.set()
                self._resources_safe = False

    def _conversation_loop(self) -> None:
        while True:
            item = self._transcripts.get()
            try:
                if item is _STOP_WORKER:
                    return

                finished_item = self._wait_for_end_of_turn(item)

                if finished_item is not None and not self._stop_event.is_set():
                    self.handle_transcript(
                        finished_item.text,
                        epoch=finished_item.epoch,
                        submitted_at=finished_item.submitted_at,
                    )
            finally:
                self._transcripts.task_done()

    def _wait_for_end_of_turn(
        self, item: _QueuedTranscript
    ) -> _QueuedTranscript | None:
        """
        Moonshine ends a line at short pauses, so one sentence can arrive as several
        transcripts. Transcripts that complete during the wait are joined. Returns
        None when the worker must stop.
        """

        while not self._stop_event.is_set():
            with self._transcript_lock:
                try:
                    newer = self._transcripts.get_nowait()
                except queue.Empty:
                    newer = None
                else:
                    self._transcripts.task_done()

                    if newer is _STOP_WORKER:
                        self._transcripts.put_nowait(newer)
                        return None

                line_started_at = self._line_started_at
                line_ended_at = self._line_ended_at

            if newer is not None:
                item = _join_transcripts(item, newer)
                continue

            now = time.monotonic()

            if line_started_at is None:
                if now - line_ended_at >= self._end_of_turn_delay:
                    return item
            elif now - line_started_at >= self._recognition_timeout:
                return item  # A line that never completes must not hold the answer.

            self._stop_event.wait(_END_OF_TURN_POLL_SECONDS)

        return None

    def _stop_conversation_worker(self) -> None:
        self._watchdog_stop.set()
        watchdog = self._watchdog_thread

        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=2.0)

        self._watchdog_thread = None
        thread = self._conversation_thread
        if thread is None:
            return

        with self._transcript_lock:
            self._drop_pending_transcript()
            try:
                self._transcripts.put_nowait(_STOP_WORKER)
            except queue.Full:
                pass  # A stop request is already queued.

        thread.join(timeout=5.0)

        if thread.is_alive():
            self._report_error("conversation worker did not stop promptly")
        else:
            self._conversation_thread = None

    def run(self) -> None:
        agent = self.build_agent()
        try:
            print("Loading local speech models...", file=self._output, flush=True)
            agent.load()
            self._configure_microphone(agent)
            self._enable_live_barge_in(agent)
            self._start_conversation_worker()
            agent.start_listening()
            print(
                f"Local microphone: {self._input_device_description(agent)}",
                file=self._output,
                flush=True,
            )
            print(
                "Ready. Speak naturally, including to interrupt; press Ctrl-C to stop.",
                file=self._output,
                flush=True,
            )
            self._print_phase("listening")
            self._stop_event.wait()
        except KeyboardInterrupt:
            print("\nStopping...", file=self._output, flush=True)
        finally:
            self.request_stop()
            try:
                agent.stop_listening()
            finally:
                self._stop_conversation_worker()

                if self.can_close_resources:
                    agent.close()
                else:
                    self._report_error(
                        "native workers remain active; resources left for process exit"
                    )

    def request_stop(self) -> None:
        self._stop_event.set()
        self._recognition_deadline.clear()
        self._response_deadline.clear()
        self._interrupt_current_reply()

    def _input_device_description(self, agent: Any) -> str:
        import sounddevice

        # After start_listening() the open stream reports the device it uses,
        # also when the system default was chosen.
        selected = agent._mic._sd_stream.device
        device = sounddevice.query_devices(selected, "input")
        return f"{device['name']} [{selected}]"

    def _print_phase(self, phase: str) -> None:
        print(f"[Local: {phase}]", file=self._output, flush=True)

    def _report_progress(self, fraction: float, name: str) -> None:
        clamped_fraction = max(0.0, min(1.0, fraction))
        percent = clamped_fraction * 100
        print(
            f"Loading speech assets: {percent:5.1f}% {name}",
            file=self._output,
            flush=True,
        )

    def _report_error(self, error: object) -> None:
        print(f"Voice chat error: {error}", file=self._error_output, flush=True)
