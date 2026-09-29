from __future__ import annotations

import base64
import io
import json
import unittest
from collections.abc import Callable
from unittest import mock

import numpy as np
from fakes import FakeOnlineLookup

from offline_voice_chat.audio import (
    PlaybackPosition,
)
from offline_voice_chat.realtime import RealtimeVoiceApplication, build_session_update

# A watchdog time far past every Realtime timeout.
FAR_FUTURE = 10**9
THREAD_CLASS = "offline_voice_chat.realtime.threading.Thread"
SESSION_HANDSHAKE = ({"type": "session.created"}, {"type": "session.updated"})
SPEECH_STARTED = {"type": "input_audio_buffer.speech_started"}
SPEECH_STOPPED = {"type": "input_audio_buffer.speech_stopped"}


def committed(item_id: str) -> dict[str, object]:
    return {"type": "input_audio_buffer.committed", "item_id": item_id}


def transcript_done(item_id: str, transcript: str) -> dict[str, object]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "transcript": transcript,
    }


def response_created(response_id: str) -> dict[str, object]:
    return {"type": "response.created", "response": {"id": response_id}}


def response_done(
    response_id: str,
    status: str = "completed",
    *,
    output: list[dict[str, object]] | None = None,
    status_details: dict[str, object] | None = None,
) -> dict[str, object]:
    response = {
        "id": response_id,
        "status": status,
        "status_details": status_details,
        "output": output or [],
    }
    return {"type": "response.done", "response": response}


def audio_delta(response_id: str, item_id: str, pcm16: bytes) -> dict[str, object]:
    return {
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "item_id": item_id,
        "content_index": 0,
        "delta": base64.b64encode(pcm16).decode("ascii"),
    }


def transcript_delta(response_id: str, text: str) -> dict[str, object]:
    return {
        "type": "response.output_audio_transcript.delta",
        "response_id": response_id,
        "delta": text,
    }


def run_started_thread(thread_class: mock.Mock) -> None:
    """Run the target that the code under test handed to a mocked Thread."""

    options = thread_class.call_args.kwargs
    target = options["target"]
    target(*options["args"])


class FakeSocket:
    def __init__(self, events: tuple[dict[str, object], ...] = ()) -> None:
        self.sent: list[dict[str, object]] = []
        self.events = list(events)
        self.timeout: float | None = None
        self.closed = False

    def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    def recv(self) -> str:
        return json.dumps(self.events.pop(0))

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def ping(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def sent_types(self) -> list[str]:
        return [event["type"] for event in self.sent]

    def reply_requests(self) -> list[dict[str, object]]:
        return [event for event in self.sent if event["type"] == "response.create"]


class FakeConnector:
    """Hand prepared sockets to the application's websocket_factory."""

    def __init__(self) -> None:
        self.sockets: list[FakeSocket] = []
        self.urls: list[str] = []
        self.options: list[dict[str, object]] = []
        self.before_connect: Callable[[], None] | None = None
        self.error: Exception | None = None

    def connect(self, url: str, **options: object) -> FakeSocket:
        self.urls.append(url)
        self.options.append(options)

        if self.before_connect is not None:
            self.before_connect()

        if self.error is not None:
            raise self.error

        return self.sockets.pop(0)


class FakePlayback:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, int, bytes]] = []

    def clear(self) -> PlaybackPosition:
        return PlaybackPosition("assistant-item", 0, 420)

    def enqueue(self, item_id: str, content_index: int, pcm16: bytes) -> None:
        self.enqueued.append((item_id, content_index, pcm16))

    def finish_item(self, item_id: str, content_index: int) -> None:
        pass

    def health_error(self, *, stall_timeout: float, now: float | None) -> None:
        return None


class FakeStoryReader:
    def __init__(self, text: str = "Goodnight, little hedgehog.") -> None:
        self.text = text

    def read(self) -> str:
        return self.text


class RealtimeConnectionTests(unittest.TestCase):
    def test_builds_audio_session_with_low_eagerness_turn_detection(self) -> None:
        event = build_session_update(
            model="gpt-realtime-2.1",
            voice="marin",
            instructions="Be kind.",
        )

        self.assertEqual(event["type"], "session.update")
        session = event["session"]
        self.assertEqual(session["output_modalities"], ["audio"])
        self.assertEqual(session["reasoning"], {"effort": "low"})
        self.assertEqual(session["tool_choice"], "auto")
        # The story reader is the only local tool Realtime may call.
        tool_signatures = [
            (tool["name"], tool["parameters"]["properties"])
            for tool in session["tools"]
        ]
        self.assertEqual(tool_signatures, [("read_goodnight_story", {})])
        audio_input = session["audio"]["input"]
        self.assertEqual(audio_input["format"]["rate"], 24_000)
        self.assertEqual(audio_input["transcription"], {"model": "gpt-transcribe"})
        turn_detection = audio_input["turn_detection"]
        self.assertEqual(turn_detection["type"], "semantic_vad")
        self.assertEqual(turn_detection["eagerness"], "low")
        self.assertNotIn("silence_duration_ms", turn_detection)
        self.assertFalse(turn_detection["create_response"])
        self.assertTrue(turn_detection["interrupt_response"])


class RealtimeEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = io.StringIO()
        self.errors = io.StringIO()
        self.socket = FakeSocket()
        self.connector = FakeConnector()
        self.playback = FakePlayback()
        self.online_lookup = FakeOnlineLookup()
        self.story_reader = FakeStoryReader()
        self.application = self.make_application()
        self.application._socket = self.socket
        self.application._playback = self.playback

    def make_application(self, **options: object) -> RealtimeVoiceApplication:
        return RealtimeVoiceApplication(
            api_key="not-used",
            model="gpt-realtime-2.1",
            voice="marin",
            instructions="Be kind.",
            story_reader=self.story_reader.read,
            websocket_factory=self.connector.connect,
            online_lookup=self.online_lookup.check,
            sounddevice_module=object(),
            numpy_module=np,
            output=self.output,
            error_output=self.errors,
            **options,
        )

    def receive(self, *events: dict[str, object]) -> None:
        for event in events:
            self.application._handle_server_event(event)

    def receive_and_run_worker(self, event: dict[str, object]) -> None:
        """Handle an event that starts a worker thread, then run that worker."""

        with mock.patch(THREAD_CLASS) as thread_class:
            self.receive(event)
            run_started_thread(thread_class)

    def complete_tool_call(self, name: str) -> None:
        function_call = {
            "type": "function_call",
            "name": name,
            "call_id": f"{name}-call",
            "arguments": "{}",
        }
        self.receive(response_created(name))
        self.receive_and_run_worker(response_done(name, output=[function_call]))

    def sent_tool_result(self) -> str:
        tool_output = self.socket.sent[0]["item"]["output"]
        return json.loads(tool_output)["result"]

    def test_story_function_call_returns_file_text_for_speech(self) -> None:
        self.application._connection_ready.set()
        self.complete_tool_call("read_goodnight_story")

        self.assertIn("reading story", self.output.getvalue())
        self.assertEqual(self.sent_tool_result(), "Goodnight, little hedgehog.")
        self.assertEqual(
            self.socket.sent[1],
            {"type": "response.create", "response": {"tool_choice": "none"}},
        )

    def test_online_transcript_routes_exact_query_before_model_reply(self) -> None:
        self.application._connection_ready.set()
        self.receive(SPEECH_STARTED, SPEECH_STOPPED, committed("user-1"))
        self.assertEqual(self.socket.reply_requests(), [])

        with mock.patch(THREAD_CLASS) as thread_class:
            self.receive(
                transcript_done("user-1", "Please, check online: news in Switzerland!")
            )
            thread_class.assert_called_once()
            run_started_thread(thread_class)

        self.assertEqual(self.online_lookup.queries, ["news in Switzerland"])
        replies = self.socket.reply_requests()
        self.assertEqual(len(replies), 1)
        reply = replies[0]["response"]
        self.assertEqual(reply["input"], [])
        self.assertEqual(reply["tool_choice"], "none")
        self.assertIn("Swiss news result.", reply["instructions"])

    def test_bare_online_prefix_waits_for_next_spoken_query(self) -> None:
        self.application._connection_ready.set()
        self.receive(
            committed("prefix"), transcript_done("prefix", "Please check online.")
        )
        self.assertEqual(self.online_lookup.queries, [])
        self.assertEqual(self.socket.reply_requests(), [])
        self.application._check_active_turn_health(now=FAR_FUTURE)
        self.assertNotIn("reconnecting", self.output.getvalue())

        self.receive(SPEECH_STARTED, SPEECH_STOPPED, committed("query"))
        self.receive_and_run_worker(transcript_done("query", "news in Switzerland."))

        self.assertEqual(self.online_lookup.queries, ["news in Switzerland"])
        last_reply = self.socket.sent[-1]["response"]
        self.assertIn("Swiss news result.", last_reply["instructions"])

    def test_other_transcript_starts_a_model_reply(self) -> None:
        self.application._connection_ready.set()
        self.receive(
            committed("user-1"), transcript_done("user-1", "Read the goodnight story")
        )
        self.assertEqual(self.socket.sent, [{"type": "response.create"}])
        self.assertEqual(self.online_lookup.queries, [])

    def test_speech_started_stops_playback_and_truncates_heard_audio(self) -> None:
        self.receive(
            response_created("answer"),
            transcript_delta("answer", "Hello"),
            SPEECH_STARTED,
        )

        self.assertEqual(
            self.socket.sent,
            [
                {
                    "type": "conversation.item.truncate",
                    "item_id": "assistant-item",
                    "content_index": 0,
                    "audio_end_ms": 420,
                }
            ],
        )
        self.assertIn("[interrupted]", self.output.getvalue())
        self.assertIn("You: [speaking]", self.output.getvalue())

    def test_response_requested_before_new_speech_is_cancelled_not_played(
        self,
    ) -> None:
        self.application._connection_ready.set()
        new_pcm = b"\x03\x00\x04\x00"

        # The first question's transcript requests a response.
        self.receive(committed("user-1"), transcript_done("user-1", "What is a GPU?"))

        # The user asks again before the server creates that response.
        self.receive(
            SPEECH_STARTED,
            SPEECH_STOPPED,
            response_created("old"),
            audio_delta("old", "old-item", b"\x01\x00\x02\x00"),
        )

        self.assertEqual(self.playback.enqueued, [])
        self.assertIn(
            {"type": "response.cancel", "response_id": "old"}, self.socket.sent
        )

        # The second question gets its own response, which plays.
        self.receive(
            committed("user-2"),
            transcript_done("user-2", "What time is it?"),
            response_created("new"),
            audio_delta("new", "new-item", new_pcm),
        )

        self.assertEqual(self.playback.enqueued, [("new-item", 0, new_pcm)])
        self.assertEqual(
            self.socket.reply_requests(),
            [{"type": "response.create"}, {"type": "response.create"}],
        )

    def test_reconnect_reconfigures_a_fresh_session(self) -> None:
        new_socket = FakeSocket(SESSION_HANDSHAKE)
        self.connector.sockets.append(new_socket)
        self.application._connection_ready.set()
        self.application._request_reconnect("test disconnect")

        recovered = self.application._recover_connection()
        self.receive(committed("user-1"), transcript_done("user-1", "Hello"))

        self.assertTrue(recovered)
        self.assertTrue(self.socket.closed)
        self.assertEqual(new_socket.sent_types(), ["session.update", "response.create"])
        self.assertIn("context restarted", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
