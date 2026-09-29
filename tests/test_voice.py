from __future__ import annotations

import io
import sys
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from fakes import (
    FakeMic,
    FakeSounddevice,
    FakeTTS,
    patch_moonshine_modules,
)

from offline_voice_chat.voice import VoiceChatApplication

# A watchdog time far past every local timeout.
FAR_FUTURE = 10**9


class FakeAgent:
    """Stand-in for Moonshine's AgentFlow after load()."""

    def __init__(
        self,
        *,
        tts: FakeTTS | None = None,
        load_error: Exception | None = None,
    ) -> None:
        self.settings: dict[str, object] = {}
        self.handler = None
        self.loaded = False
        self.started = False
        self.listening = threading.Event()
        self.stopped_listening = False
        self.closed = False
        self.load_error = load_error
        self._mic = FakeMic()
        self._tts = FakeTTS() if tts is None else tts

    def language(self, value: str):
        self.settings["language"] = value
        return self

    def use_embeddings(self, value: bool):
        self.settings["use_embeddings"] = value
        return self

    def barge_in(self, value: bool):
        self.settings["barge_in"] = value
        return self

    def beeps(self, value: bool):
        self.settings["beeps"] = value
        return self

    def on_progress(self, callback):
        self.settings["on_progress"] = callback
        return self

    def on_error(self, callback):
        self.settings["on_error"] = callback
        return self

    def otherwise(self, handler):
        self.handler = handler
        return self

    def voice(self, value: str):
        self.settings["voice"] = value
        return self

    def output_device(self, value):
        self.settings["output_device"] = value
        return self

    def models_from(self, value: Path):
        self.settings["models_from"] = value
        return self

    def load(self) -> None:
        self.loaded = True
        if self.load_error is not None:
            raise self.load_error

    def start_listening(self) -> None:
        self.started = True
        self.listening.set()

    def stop_listening(self) -> None:
        self.stopped_listening = True

    def close(self) -> None:
        self.closed = True


class RecordingModel:
    """A stream_reply stand-in that records transcripts and answers at once."""

    def __init__(self, *parts: str) -> None:
        self.parts = parts
        self.transcripts: list[str] = []

    def stream_reply(self, transcript: str):
        self.transcripts.append(transcript)
        yield from self.parts


class SlowFirstReplyModel:
    """Think about the first request until the test releases it."""

    def __init__(self) -> None:
        self.transcripts: list[str] = []
        self.first_request_started = threading.Event()
        self.release_first_request = threading.Event()

    def stream_reply(self, transcript: str):
        self.transcripts.append(transcript)

        if len(self.transcripts) == 1:
            self.first_request_started.set()
            self.release_first_request.wait(timeout=2)

        yield "It will be sunny."


class TurnLog:
    """Record finalize_turn results and let a test wait for them."""

    def __init__(self) -> None:
        self.delivered: list[bool] = []
        self._changed = threading.Condition()

    def record(self, delivered: bool) -> None:
        with self._changed:
            self.delivered.append(delivered)
            self._changed.notify_all()

    def wait_for(self, count: int) -> bool:
        with self._changed:
            return self._changed.wait_for(
                lambda: len(self.delivered) >= count, timeout=2
            )

    def wait_for_delivered_turn(self) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: True in self.delivered, timeout=2)


@dataclass
class RunningVoiceChat:
    application: VoiceChatApplication
    agent: FakeAgent
    mic: FakeMic
    tts: FakeTTS
    turns: TurnLog
    runner: threading.Thread


class VoiceChatApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = io.StringIO()
        self.errors = io.StringIO()
        self.sounddevice = FakeSounddevice({7: "USB microphone"})
        moonshine_modules = patch_moonshine_modules()
        moonshine_modules.start()
        self.addCleanup(moonshine_modules.stop)
        sounddevice_module = mock.patch.dict(
            sys.modules, {"sounddevice": self.sounddevice}
        )
        sounddevice_module.start()
        self.addCleanup(sounddevice_module.stop)

    def make_application(
        self, stream_reply, agent: FakeAgent, **options: object
    ) -> VoiceChatApplication:
        return VoiceChatApplication(
            stream_reply,
            language="en",
            agent_factory=lambda: agent,
            output=self.output,
            error_output=self.errors,
            **options,
        )

    def start_voice(
        self, stream_reply, *, end_of_turn_delay: float = 0.0, **options: object
    ) -> RunningVoiceChat:
        """Run the application on a thread with a fake microphone and speaker."""

        agent = FakeAgent(tts=FakeTTS(hold_playback=True))
        turns = TurnLog()
        application = self.make_application(
            stream_reply,
            agent,
            end_of_turn_delay=end_of_turn_delay,
            finalize_turn=turns.record,
            **options,
        )
        runner = threading.Thread(target=application.run, daemon=True)
        runner.start()
        self.assertTrue(agent.listening.wait(timeout=1))

        def stop() -> None:
            application.request_stop()
            runner.join(timeout=5)

        self.addCleanup(stop)
        return RunningVoiceChat(
            application=application,
            agent=agent,
            mic=agent._mic,
            tts=agent._tts,
            turns=turns,
            runner=runner,
        )

    def test_stop_command_ends_the_answer_without_model_inference(self) -> None:
        model = RecordingModel("Answer.")
        voice = self.start_voice(model.stream_reply)
        voice.application.submit_transcript("question")
        self.assertTrue(voice.tts.playback_started.wait(timeout=1))

        voice.application.submit_transcript("Please stop!")

        self.assertTrue(voice.turns.wait_for(1))
        self.assertEqual(voice.turns.delivered, [False])
        self.assertEqual(model.transcripts, ["question"])
        self.assertIn("[interrupted]", self.output.getvalue())
        self.assertNotIn("Stopping...", self.output.getvalue())
        self.assertTrue(voice.runner.is_alive())

    def test_terminate_voice_command_closes_the_application(self) -> None:
        model = RecordingModel()
        voice = self.start_voice(model.stream_reply)

        voice.application.submit_transcript("Please terminate yourself!")
        voice.runner.join(timeout=2)

        self.assertFalse(voice.runner.is_alive())
        self.assertEqual(model.transcripts, [])
        self.assertTrue(voice.agent.stopped_listening)
        self.assertTrue(voice.agent.closed)
        self.assertIn("Stopping...", self.output.getvalue())

    def test_prints_and_speaks_streamed_answer(self) -> None:
        agent = FakeAgent()
        finalized: list[bool] = []
        application = self.make_application(
            RecordingModel("Hello", " there").stream_reply,
            agent,
            finalize_turn=finalized.append,
        )

        application.handle_transcript("Hi")

        self.assertEqual(agent._tts.parts, ["Hello", " there"])
        self.assertEqual(finalized, [True])
        self.assertIn("You: Hi", self.output.getvalue())
        self.assertIn("Assistant: Hello there", self.output.getvalue())

    def test_run_loads_starts_and_always_closes_agent(self) -> None:
        agent = FakeAgent()
        application = self.make_application(RecordingModel().stream_reply, agent)
        application.request_stop()

        application.run()

        self.assertTrue(agent.loaded)
        self.assertTrue(agent.started)
        self.assertTrue(agent.closed)

    def test_first_non_echo_partial_stops_active_reply(self) -> None:
        model = RecordingModel("The answer being spoken.")
        voice = self.start_voice(model.stream_reply)
        voice.application.submit_transcript("question")
        self.assertTrue(voice.tts.playback_started.wait(timeout=1))

        # A line onset alone may be speaker bleed, so playback continues.
        voice.mic.start_line()
        self.assertEqual(voice.turns.delivered, [])
        voice.mic.change_line_text("Please stop")

        self.assertTrue(voice.turns.wait_for(1))
        self.assertEqual(voice.turns.delivered, [False])
        self.assertIn("[interrupted]", self.output.getvalue())
        self.assertEqual(voice.tts.cancel_count, 1)
        # Moonshine's pending CANCELLED result is drained before the next reply.
        self.assertFalse(voice.tts.cancel_pending)

        # The interrupted answer is kept as the echo reference.
        voice.application.submit_transcript("The answer being spoken.")
        self.assertIn("Ignored probable speaker echo", self.output.getvalue())
        self.assertEqual(model.transcripts, ["question"])

    def test_probable_partial_speaker_echo_does_not_stop_playback(self) -> None:
        answer = "The answer being spoken through the speakers is still going."
        model = RecordingModel(answer)
        voice = self.start_voice(model.stream_reply)
        voice.application.submit_transcript("question")
        self.assertTrue(voice.tts.playback_started.wait(timeout=1))

        voice.mic.start_line()
        voice.mic.change_line_text("The answer being spoken")
        voice.application.submit_transcript(
            "The answer being spoken through the speakers"
        )
        voice.mic.complete_line("The answer being spoken through the speakers")
        voice.tts.release_playback.set()

        self.assertTrue(voice.turns.wait_for(1))
        self.assertEqual(voice.turns.delivered, [True])
        self.assertEqual(model.transcripts, ["question"])
        self.assertIn("Ignored probable speaker echo", self.output.getvalue())

    def test_interruption_queues_and_speaks_the_follow_up(self) -> None:
        release_generation = threading.Event()
        self.addCleanup(release_generation.set)
        transcripts: list[str] = []

        def reply(text: str):
            transcripts.append(text)

            if text == "first question":
                yield "First answer."
                release_generation.wait(timeout=2)
                yield " This must not be spoken."
                return

            yield "Second answer."

        voice = self.start_voice(reply)
        voice.tts.release_playback.set()
        voice.application.submit_transcript("first question")
        self.assertTrue(voice.tts.first_push.wait(timeout=1))

        voice.application.submit_transcript("a different follow-up question")
        release_generation.set()

        self.assertTrue(voice.turns.wait_for(2))
        self.assertEqual(
            transcripts, ["first question", "a different follow-up question"]
        )
        self.assertGreaterEqual(voice.tts.cancel_count, 1)
        self.assertIn("First answer.", voice.tts.parts)
        self.assertNotIn(" This must not be spoken.", voice.tts.parts)
        self.assertIn("Second answer.", voice.tts.parts)
        self.assertIn("[interrupted]", self.output.getvalue())
        self.assertEqual(voice.turns.delivered, [False, True])

    def start_slow_first_reply(
        self, *, end_of_turn_delay: float
    ) -> tuple[RunningVoiceChat, SlowFirstReplyModel]:
        model = SlowFirstReplyModel()
        self.addCleanup(model.release_first_request.set)
        voice = self.start_voice(
            model.stream_reply, end_of_turn_delay=end_of_turn_delay
        )
        voice.tts.release_playback.set()
        return voice, model

    def test_a_sentence_split_at_a_short_pause_is_answered_as_one_question(
        self,
    ) -> None:
        voice, model = self.start_slow_first_reply(end_of_turn_delay=0.3)
        model.release_first_request.set()

        # Moonshine closes a line at the pause, and AgentFlow submits each line.
        for line in ("What is the weather like?", "In Zurich tomorrow?"):
            voice.mic.start_line()
            voice.application.submit_transcript(line)
            voice.mic.complete_line(line)

        self.assertTrue(voice.turns.wait_for_delivered_turn())
        self.assertEqual(
            model.transcripts, ["What is the weather like? In Zurich tomorrow?"]
        )


if __name__ == "__main__":
    unittest.main()
