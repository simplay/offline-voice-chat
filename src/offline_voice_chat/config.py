"""Command-line configuration for the voice chatbot."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Sequence

from .llama import MIN_PROMPT_TOKENS
from .memory import default_memory_path
from .system_prompts import (
    default_system_prompts_dir,
    load_default_system_prompt,
    load_system_prompt,
)


@dataclass(frozen=True)
class AppConfig:
    model_path: Path | None
    backend: str = "local"
    language: str = "en"
    voice: str | None = None
    output_device: int | str | None = None
    models_dir: Path | None = None
    files_dir: Path | None = None
    context_size: int = 4096
    max_tokens: int = 256
    temperature: float = 0.4
    threads: int | None = None
    gpu_layers: int = 0
    max_history_turns: int = 8
    system_prompt: str = field(default_factory=load_default_system_prompt)
    speech_update_interval: float = 0.25
    end_of_turn_delay: float = 1.0
    input_device: int | str | None = None
    realtime_model: str = "gpt-realtime-2.1"
    realtime_voice: str = "marin"
    realtime_api_key_env: str = "OPENAI_API_KEY"
    persistent_memory: bool = True
    memory_file: Path = field(default_factory=default_memory_path)
    clear_memory: bool = False
    report_latency: bool = False
    select_audio_devices: bool = False
    menu: bool = False
    system_prompts_dir: Path = field(default_factory=default_system_prompts_dir)
    system_prompt_file: Path | None = None
    echo_cancel: bool = False

    def with_audio_devices(
        self, input_device: int | str | None, output_device: int | str | None
    ) -> AppConfig:
        return replace(self, input_device=input_device, output_device=output_device)


# argparse reads its defaults from AppConfig so the two cannot drift apart.
_DEFAULTS: dict[str, Any] = {item.name: item.default for item in fields(AppConfig)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="offline-voice-chat",
        description=(
            "Talk to a local GGUF model, or explicitly opt in to an OpenAI "
            "Realtime speech-to-speech session."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("local", "realtime"),
        default=_DEFAULTS["backend"],
        help="Voice backend (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to a local instruction-tuned GGUF model (local backend).",
    )
    parser.add_argument(
        "--language",
        default=_DEFAULTS["language"],
        help="Moonshine speech language, for example 'en' or 'de'.",
    )
    parser.add_argument(
        "--voice",
        help="Optional Moonshine catalog voice such as kokoro_af_heart.",
    )
    parser.add_argument(
        "--output-device",
        help=(
            "Optional PortAudio output-device index or name; omit it to use "
            "the system default."
        ),
    )
    parser.add_argument(
        "--input-device",
        help=(
            "Optional PortAudio input-device index or name; omit it to use "
            "the system default."
        ),
    )
    parser.add_argument(
        "--select-audio-devices",
        action="store_true",
        help=(
            "Choose a microphone and audio output before starting; "
            "with --menu, use Configure audio instead."
        ),
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Show the startup menu (always on in the Linux launcher).",
    )
    parser.add_argument(
        "--echo-cancel",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS["echo_cancel"],
        help=(
            "Use Linux desktop acoustic echo cancellation with pulse audio devices "
            "(enabled by the Linux launcher; --no-echo-cancel disables it)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Optional directory for Moonshine speech assets.",
    )
    parser.add_argument(
        "--files-dir",
        type=Path,
        help=(
            "Optional directory whose top-level UTF-8 .txt files may be read "
            "when explicitly requested."
        ),
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=_DEFAULTS["context_size"],
        help="llama.cpp context window in tokens (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULTS["max_tokens"],
        help="Maximum generated tokens per answer (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_DEFAULTS["temperature"],
        help="Sampling temperature from 0 through 2 (default: %(default)s).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU inference threads; omit to let llama.cpp choose.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=_DEFAULTS["gpu_layers"],
        help=(
            "Layers to offload; use -1 for all possible layers (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=_DEFAULTS["max_history_turns"],
        help=(
            "Full user/assistant turns retained in active-session working "
            "context (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--persistent-memory",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS["persistent_memory"],
        help=(
            "Keep a compact local summary across restarts; use "
            "--no-persistent-memory for session-only history (default: enabled)."
        ),
    )
    parser.add_argument(
        "--memory-file",
        type=Path,
        help=(
            "Persistent-memory JSON path; defaults to the per-user local "
            "application-data directory."
        ),
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Delete stored persistent memory before starting.",
    )
    prompt_options = parser.add_mutually_exclusive_group()
    prompt_options.add_argument(
        "--system-prompt",
        help="Use this prompt text instead of loading a system prompt file.",
    )
    prompt_options.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Read a UTF-8 system prompt from this file.",
    )
    parser.add_argument(
        "--system-prompts-dir",
        type=Path,
        default=default_system_prompts_dir(),
        help="Directory of prompt files (default: the checkout's system-prompts).",
    )
    parser.add_argument(
        "--speech-update-interval",
        type=float,
        default=_DEFAULTS["speech_update_interval"],
        help=(
            "Minimum seconds between streaming speech-recognition updates "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--end-of-turn-delay",
        type=float,
        default=_DEFAULTS["end_of_turn_delay"],
        help=(
            "Seconds of silence after a recognized line before the assistant "
            "answers; speech in that time joins the same question "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Report local per-turn queue, first-text, and first-audio-write latency.",
    )
    parser.add_argument(
        "--realtime-model",
        default=_DEFAULTS["realtime_model"],
        help=(
            "OpenAI Realtime model used only by --backend realtime "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--realtime-voice",
        default=_DEFAULTS["realtime_voice"],
        help="OpenAI Realtime output voice (default: %(default)s).",
    )
    parser.add_argument(
        "--realtime-api-key-env",
        default=_DEFAULTS["realtime_api_key_env"],
        help=(
            "Environment variable containing the OpenAI API key; the key "
            "itself is never accepted as an argument (default: %(default)s)."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_path = None

    if args.model is not None:
        model_path = args.model.expanduser().resolve()

        if not model_path.is_file():
            parser.error(f"GGUF model does not exist or is not a file: {model_path}")

        if model_path.suffix.lower() != ".gguf":
            parser.error(f"model must be a .gguf file: {model_path}")
    elif args.backend == "local":
        parser.error("--model is required when --backend local is selected")

    language = args.language.strip()

    if not language:
        parser.error("language must not be empty")

    if args.context_size < 512:
        parser.error("context-size must be at least 512")

    if args.max_tokens < 1:
        parser.error("max-tokens must be at least 1")

    if args.context_size - args.max_tokens < MIN_PROMPT_TOKENS:
        parser.error(
            f"context-size must leave at least {MIN_PROMPT_TOKENS} tokens "
            "beyond max-tokens"
        )

    if not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")

    if args.threads is not None and args.threads < 1:
        parser.error("threads must be at least 1")

    if args.gpu_layers < -1:
        parser.error("gpu-layers must be -1 or greater")

    if args.max_history_turns < 1:
        parser.error("max-history-turns must be at least 1")

    if (
        not math.isfinite(args.speech_update_interval)
        or args.speech_update_interval < 0.05
    ):
        parser.error("speech-update-interval must be at least 0.05")

    # A chained comparison is False for NaN and infinity as well.
    if not 0 <= args.end_of_turn_delay <= 10:
        parser.error("end-of-turn-delay must be between 0 and 10 seconds")

    if not args.realtime_model.strip():
        parser.error("realtime-model must not be empty")

    if not args.realtime_voice.strip():
        parser.error("realtime-voice must not be empty")

    if not args.realtime_api_key_env.strip():
        parser.error("realtime-api-key-env must not be empty")

    models_dir = None

    if args.models_dir is not None:
        models_dir = args.models_dir.expanduser().resolve()

    files_dir = None

    if args.files_dir is not None:
        files_dir = args.files_dir.expanduser().resolve()

        if not files_dir.is_dir():
            parser.error(f"files-dir does not exist or is not a directory: {files_dir}")

    output_device = _parse_device(args.output_device)
    input_device = _parse_device(args.input_device)
    system_prompts_dir = args.system_prompts_dir.expanduser().resolve()
    system_prompt_file = None

    if args.system_prompt is not None:
        system_prompt = args.system_prompt.strip()

        if not system_prompt:
            parser.error("system-prompt must not be empty")
    else:
        system_prompt_file = system_prompts_dir / "default.txt"

        if args.system_prompt_file is not None:
            system_prompt_file = args.system_prompt_file.expanduser().resolve()

        try:
            system_prompt = load_system_prompt(system_prompt_file)
        except ValueError as exception:
            parser.error(str(exception))

    memory_file = default_memory_path().resolve()

    if args.memory_file is not None:
        memory_file = args.memory_file.expanduser().resolve()

    return AppConfig(
        model_path=model_path,
        backend=args.backend,
        language=language,
        voice=args.voice.strip() if args.voice else None,
        output_device=output_device,
        models_dir=models_dir,
        files_dir=files_dir,
        context_size=args.context_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        max_history_turns=args.max_history_turns,
        system_prompt=system_prompt,
        speech_update_interval=args.speech_update_interval,
        end_of_turn_delay=args.end_of_turn_delay,
        input_device=input_device,
        realtime_model=args.realtime_model.strip(),
        realtime_voice=args.realtime_voice.strip(),
        realtime_api_key_env=args.realtime_api_key_env.strip(),
        persistent_memory=args.persistent_memory,
        memory_file=memory_file,
        clear_memory=args.clear_memory,
        report_latency=args.latency,
        select_audio_devices=args.select_audio_devices,
        menu=args.menu,
        system_prompts_dir=system_prompts_dir,
        system_prompt_file=system_prompt_file,
        echo_cancel=args.echo_cancel,
    )


def _parse_device(value: str | None) -> int | str | None:
    if not value:
        return None

    text = value.strip()
    return int(text) if text.isdigit() else text
