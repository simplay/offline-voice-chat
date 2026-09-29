from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from offline_voice_chat.llama import LlamaCppChatModel


class FakeStreamingModel:
    """Answer each request with the next list of text parts."""

    prompt_budget = 3584

    def __init__(self, replies: list[list[str]]) -> None:
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []
        self.closed = False

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def prepare_reply(self) -> None:
        pass

    def cancel_current_reply(self) -> None:
        pass

    def stream_reply(self, messages: list[dict[str, str]]):
        self.calls.append([message.copy() for message in messages])
        parts = self.replies.pop(0)

        try:
            yield from parts
        except GeneratorExit:
            self.closed = True
            raise


class FakeContextTool:
    def __init__(self, context: str | None) -> None:
        self.context = context
        self.calls: list[str] = []
        self.reset_count = 0

    def context_for_request(self, request: str) -> str | None:
        self.calls.append(request)
        return self.context

    def reset(self) -> None:
        self.reset_count += 1


class FakeMemory:
    def __init__(self, context: str | None = None) -> None:
        self.context = context
        self.completed_turns: list[tuple[str, str]] = []

    def context_for_chat(self) -> str | None:
        return self.context

    def record_complete_turn(self, user_text: str, assistant_text: str) -> bool:
        self.completed_turns.append((user_text, assistant_text))
        return True


class FakeOnlineLookup:
    """Record online queries and answer each with the same text."""

    def __init__(self, result: str = "Swiss news result.") -> None:
        self.result = result
        self.queries: list[str] = []

    def check(self, query: str) -> str:
        self.queries.append(query)
        return self.result


class FakeClock:
    """A monotonic clock that a test moves forward by hand."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class InstantOutputStream:
    """Accept every write at once and report a fixed output latency."""

    def __init__(self, latency: float) -> None:
        self.latency = latency

    def start(self) -> None:
        pass

    def write(self, block: object) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class InstantOutputDevice:
    def __init__(self, latency: float = 0.0) -> None:
        self.latency = latency

    def check_output_settings(self, **settings: object) -> None:
        pass

    def OutputStream(self, **settings: object) -> InstantOutputStream:
        return InstantOutputStream(self.latency)


class FakeLlama:
    """Stand-in for llama_cpp.Llama with the attributes the adapter uses."""

    def __init__(
        self,
        chunks: list[dict[str, object]] | None = None,
        *,
        template_handler: Callable[..., object] | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.calls: list[dict[str, object]] = []
        self.chat_format = "chat_template.default"
        self.chat_handler: Callable[..., object] | None = None
        self._chat_handlers: dict[str, Callable[..., object]] = {}
        self.ctx = object()

        if template_handler is not None:
            self._chat_handlers[self.chat_format] = template_handler

    def create_chat_completion(self, **options: object):
        self.calls.append(options)
        return (chunk for chunk in self.chunks)

    def tokenize(self, text: bytes, add_bos: bool) -> list[int]:
        return list(text)

    def close(self) -> None:
        pass


class FakeLlamaCpp:
    """Stand-in for the llama_cpp module."""

    def __init__(self, llama: FakeLlama, *, gpu_offload: bool = True) -> None:
        self.llama = llama
        self.gpu_offload = gpu_offload
        self.model_options: dict[str, object] = {}

    def Llama(self, **options: object) -> FakeLlama:
        self.model_options = options
        return self.llama

    def llama_supports_gpu_offload(self) -> bool:
        return self.gpu_offload

    def ggml_abort_callback(self, function: Callable[[object], bool]):
        return function

    def llama_set_abort_callback(self, context, callback, data) -> None:
        pass


def text_chunk(text: str, finish_reason: str | None = None) -> dict[str, object]:
    """Build one streamed chat chunk in llama_cpp's format."""

    delta = {"content": text} if text else {}
    return {"choices": [{"delta": delta, "finish_reason": finish_reason}]}


def load_llama_model(
    llama_cpp: FakeLlamaCpp, *, gpu_layers: int = 0
) -> LlamaCppChatModel:
    with mock.patch.dict(sys.modules, {"llama_cpp": llama_cpp}):
        return LlamaCppChatModel(
            Path("model.gguf"),
            context_size=2048,
            max_tokens=100,
            temperature=0.2,
            threads=4,
            gpu_layers=gpu_layers,
        )


class FakePortAudioError(Exception):
    pass


class FakeSounddevice:
    """Stand-in for the sounddevice module's device lookup."""

    PortAudioError = FakePortAudioError

    def __init__(self, input_names: dict[int, str] | None = None) -> None:
        self.input_names = input_names or {}

    def query_devices(self, device: int, kind: str) -> dict[str, object]:
        name = self.input_names.get(device, "Built-in microphone")
        return {"name": name, "index": device}


@dataclass
class FakeAudioChunk:
    samples: list[float]
    sample_rate: int


@dataclass
class FakePlayItem:
    data: object
    sample_rate: int
    device: object


class FakeTtsModule:
    """Stand-in for moonshine_voice.tts, which start_speech() imports."""

    _PlayItem = FakePlayItem

    @staticmethod
    def _import_say_audio_deps():
        return np, None


class FakeTranscriptEventListener:
    """Stand-in for Moonshine's listener base class."""


class FakeTranscriberModule:
    TranscriptEventListener = FakeTranscriptEventListener


def patch_moonshine_modules():
    """Let the Moonshine adapter import its dependency without the package."""

    return mock.patch.dict(
        sys.modules,
        {
            "moonshine_voice.tts": FakeTtsModule(),
            "moonshine_voice.transcriber": FakeTranscriberModule(),
        },
    )


class FakeOutputStream:
    latency = 0.0

    def __init__(self) -> None:
        self.writes: list[object] = []

    def write(self, data: object) -> None:
        self.writes.append(data)


class FakeTTS:
    """Stand-in for Moonshine 0.1.5's TextToSpeech.

    It synthesizes one audio chunk per pushed text. wait() plays like
    Moonshine's drain: it writes once, then holds until the test releases
    playback or a stop is signalled.
    """

    def __init__(self, *, hold_playback: bool = False) -> None:
        self.parts: list[str] = []
        self.first_push = threading.Event()
        self.playback_started = threading.Event()
        self.release_playback = threading.Event()
        self.cancel_entered = threading.Event()
        self.cancel_gate: threading.Event | None = None
        self.cancel_count = 0
        self.cancel_pending = False
        self.is_streaming = False
        self.output_stream = FakeOutputStream()
        self._unsynthesized: list[str] = []
        self._input_ended = False
        # Moonshine's playback state, which the adapter reads directly.
        self._say_stop_event = threading.Event()
        self._say_queue: queue.Queue[object] = queue.Queue()
        self._play_queue: queue.Queue[object] = queue.Queue()
        self._synth_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._playback_tail_until = 0.0
        self._output_device = None

        if not hold_playback:
            self.release_playback.set()

    def _ensure_say_workers(self) -> None:
        self._say_stop_event.clear()
        self._input_ended = False

    def _acquire_output_stream(self, sounddevice, device, sample_rate):
        return self.output_stream

    def _play_one(self, *args: object) -> None:
        pass

    def push_text(self, text: str) -> None:
        self.parts.append(text)
        self._unsynthesized.append(text)
        self.is_streaming = True
        self.first_push.set()

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        self._input_ended = True

    def next_chunk(self) -> FakeAudioChunk | None:
        if self.cancel_pending:
            self.cancel_pending = False
            return None

        if self._unsynthesized:
            self._unsynthesized.pop(0)
            return FakeAudioChunk(samples=[0.0], sample_rate=24_000)

        if self._input_ended:
            self.is_streaming = False

        return None

    def wait(self) -> None:
        output = self._acquire_output_stream(None, None, 24_000)
        output.write([0.0])
        self.playback_started.set()

        while not self.release_playback.wait(timeout=0.01):
            if self._say_stop_event.is_set():
                return

    def cancel_stream(self) -> None:
        self.cancel_count += 1
        self.cancel_entered.set()

        if self.cancel_gate is not None:
            self.cancel_gate.wait(timeout=2)

        self.cancel_pending = True
        self.is_streaming = False
        self._unsynthesized.clear()

    def stop(self) -> None:
        self._say_stop_event.set()


@dataclass
class FakeLine:
    text: str


@dataclass
class FakeLineEvent:
    line: FakeLine


class FakeMicStream:
    def __init__(self) -> None:
        self._update_interval = 0.5


class FakeInputStream:
    def __init__(self) -> None:
        self.device = 0


class FakeMic:
    """Stand-in for Moonshine's MicTranscriber after AgentFlow.load()."""

    def __init__(self) -> None:
        self.listeners: list[object] = []
        self.mic_stream = FakeMicStream()
        self._sd_stream = FakeInputStream()

    def add_listener(self, listener: object) -> None:
        self.listeners.append(listener)

    def device(self, device: int | str) -> None:
        self._sd_stream.device = device

    def update_interval(self, seconds: float) -> None:
        pass

    def start_line(self) -> None:
        for listener in self.listeners:
            listener.on_line_started(FakeLineEvent(FakeLine("")))

    def change_line_text(self, text: str) -> None:
        for listener in self.listeners:
            listener.on_line_text_changed(FakeLineEvent(FakeLine(text)))

    def complete_line(self, text: str = "") -> None:
        for listener in self.listeners:
            listener.on_line_completed(FakeLineEvent(FakeLine(text)))
