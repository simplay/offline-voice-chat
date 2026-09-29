from __future__ import annotations

import queue
import threading
import unittest

from fakes import FakeAudioChunk, FakeOutputStream, FakeTTS, patch_moonshine_modules

from offline_voice_chat.moonshine import (
    SpeechStream,
    make_tts_interruptible,
    start_speech,
)


class FakeSynthesizer:
    def __init__(self):
        self.is_streaming = False
        self.parts = []
        self.ended = False
        self._say_stop_event = threading.Event()

    def push_text(self, text):
        self.is_streaming = True
        self.parts.append(text)

    def end_input(self):
        self.ended = True

    def next_chunk(self):
        if self.parts:
            return self.parts.pop(0)

        if self.ended:
            self.is_streaming = False

        return None

    def wait(self):
        pass

    def cancel_stream(self):
        self.is_streaming = False
        self.parts.clear()


class PlaybackQueue(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=1)
        self.capacity_wait = threading.Event()

    def put(self, item, block=True, timeout=None):
        if self.full():
            self.capacity_wait.set()

        return super().put(item, block=block, timeout=timeout)


class QueuedSynthesizer(FakeSynthesizer):
    def __init__(self):
        super().__init__()
        self._play_queue = PlaybackQueue()
        self._output_device = None

    def _ensure_say_workers(self):
        self._say_stop_event.clear()

    def next_chunk(self):
        text = super().next_chunk()
        if text is None:
            return None

        return FakeAudioChunk(samples=[0.1, 0.2], sample_rate=24000)


class PlaybackBackpressureTests(unittest.TestCase):
    def setUp(self):
        # Exercise the production queue adapter without native packages or audio.
        moonshine_modules = patch_moonshine_modules()
        moonshine_modules.start()
        self.addCleanup(moonshine_modules.stop)

    def test_cancellation_releases_a_synthesis_worker_waiting_for_capacity(self):
        synthesizer = QueuedSynthesizer()
        synthesizer._play_queue.put("already queued audio")
        errors = []
        speech = start_speech(synthesizer, errors.append)

        try:
            speech.push_text("This audio should be cancelled")
            self.assertTrue(synthesizer._play_queue.capacity_wait.wait(timeout=1))
            speech.stop()
            self.assertFalse(speech.wait())
            self.assertEqual(synthesizer._play_queue.qsize(), 1)
            self.assertEqual(errors, [])
        finally:
            speech.stop()


class SpeechStreamTests(unittest.TestCase):
    def test_synthesis_failure_is_reported_and_never_looks_delivered(self):
        class FailingSynthesizer(FakeSynthesizer):
            def next_chunk(self):
                raise RuntimeError("native synthesis failed")

        def failing_enqueue(_):
            raise RuntimeError("playback buffer full")

        for synthesizer, enqueue, message in (
            (FailingSynthesizer(), lambda _: None, "native synthesis failed"),
            (FakeSynthesizer(), failing_enqueue, "playback buffer full"),
        ):
            with self.subTest(message=message):
                errors = []
                with SpeechStream(synthesizer, enqueue, errors.append) as speech:
                    speech.push_text("Hello")
                    speech.end_input()

                    with self.assertRaisesRegex(RuntimeError, message):
                        speech.wait()

                self.assertEqual(len(errors), 1)


class MoonshineCompatibilityTests(unittest.TestCase):
    def test_interruptible_output_stops_between_short_writes(self) -> None:
        class StoppingOutputStream(FakeOutputStream):
            """Signal a stop during the first write, like a barge-in."""

            def __init__(self, stop_event: threading.Event) -> None:
                super().__init__()
                self.stop_event = stop_event

            def write(self, data) -> None:
                super().write(data)
                self.stop_event.set()

        tts = FakeTTS()
        tts.output_stream = StoppingOutputStream(tts._say_stop_event)
        activity = []

        def record_activity() -> None:
            activity.append("write")

        make_tts_interruptible(tts, record_activity)
        stream = tts._acquire_output_stream(None, None, 1000)

        stream.write(list(range(100)))

        self.assertEqual(len(tts.output_stream.writes), 1)
        self.assertEqual(len(tts.output_stream.writes[0]), 40)
        self.assertEqual(activity, ["write"])
        self.assertEqual(stream.latency, 0.0)

        # A stop drops speech that is queued but not yet played.
        tts._say_queue.put("text")
        tts._play_queue.put("audio")
        tts.stop()
        self.assertTrue(tts._say_queue.empty())
        self.assertTrue(tts._play_queue.empty())


if __name__ == "__main__":
    unittest.main()
