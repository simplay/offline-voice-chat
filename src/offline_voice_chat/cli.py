from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack

from .audio_devices import select_audio_devices
from .chat import ChatSession
from .commands import default_commands
from .config import AppConfig, parse_args
from .echo_cancellation import echo_cancelled_devices
from .file_tools import LocalTextFileReader, read_goodnight_story
from .llama import LlamaCppChatModel
from .memory import PersistentConversationMemory
from .online import OnlineLookup
from .startup_menu import configure_startup
from .voice import VoiceChatApplication


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)

    try:
        if config.menu:
            configured = configure_startup(config)
            if configured is None:
                return 0

            config = configured
        elif config.select_audio_devices:
            selection = select_audio_devices(config.input_device, config.output_device)
            if selection is None:
                return 0

            config = config.with_audio_devices(*selection)

        with ExitStack() as resources:
            if config.echo_cancel:
                devices = resources.enter_context(
                    echo_cancelled_devices(config.input_device, config.output_device)
                )
                config = config.with_audio_devices(*devices)

            memory = _prepare_memory(config)

            if config.backend == "realtime":
                _run_realtime(config)
            else:
                _run_local(config, memory)
    except (OSError, RuntimeError, ValueError, *_portaudio_errors()) as exception:
        print(f"offline-voice-chat: {exception}", file=sys.stderr)
        return 1

    return 0


def _portaudio_errors() -> tuple[type[Exception], ...]:
    """
    Return sounddevice's error type, which does not derive from OSError.
    """

    sounddevice = sys.modules.get("sounddevice")
    return () if sounddevice is None else (sounddevice.PortAudioError,)


def _prepare_memory(
    config: AppConfig,
) -> PersistentConversationMemory | None:
    if not config.persistent_memory and not config.clear_memory:
        return None

    memory = PersistentConversationMemory(
        config.memory_file,
        on_warning=lambda message: print(
            f"Persistent memory warning: {message}",
            file=sys.stderr,
            flush=True,
        ),
    )

    if config.clear_memory:
        if memory.clear():
            print(
                f"Cleared persistent memory: {config.memory_file}",
                flush=True,
            )

    if not config.persistent_memory:
        return None

    if config.backend == "realtime":
        print(
            "Persistent memory is not loaded or uploaded in Realtime mode.",
            flush=True,
        )
        return None

    if not config.clear_memory:
        memory.load()

    print(f"Persistent memory: {config.memory_file}", flush=True)
    return memory


def _run_local(
    config: AppConfig,
    memory: PersistentConversationMemory | None,
) -> None:
    print(f"Loading local model: {config.model_path}", flush=True)
    model = LlamaCppChatModel(
        config.model_path,
        context_size=config.context_size,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        threads=config.threads,
        gpu_layers=config.gpu_layers,
    )
    application: VoiceChatApplication | None = None

    try:
        if config.gpu_layers != 0 and not model.gpu_offload_available:
            print(
                "Performance warning: this llama.cpp build is CPU-only; --gpu-layers "
                "cannot enable acceleration. Install a GPU-enabled build for lower latency.",
                file=sys.stderr,
                flush=True,
            )

        if config.gpu_layers == 0:
            print("Local inference: CPU (GPU offload disabled).", flush=True)
        elif model.gpu_offload_available:
            layers = "all possible" if config.gpu_layers == -1 else config.gpu_layers
            print(
                f"Local inference: GPU offload requested for {layers} layers.",
                flush=True,
            )

        file_reader = None

        if config.files_dir is not None:
            file_reader = LocalTextFileReader(config.files_dir)

        session = ChatSession(
            model,
            system_prompt=config.system_prompt,
            max_history_turns=config.max_history_turns,
            context_tool=file_reader,
            memory=memory,
        )

        session.commands = default_commands(
            reset_conversation=session.reset_conversation,
            check_online=OnlineLookup().check,
            read_goodnight_story=lambda: read_goodnight_story(config.files_dir),
        )

        application = VoiceChatApplication(
            session.stream_reply,
            language=config.language,
            voice=config.voice,
            input_device=config.input_device,
            output_device=config.output_device,
            models_dir=config.models_dir,
            speech_update_interval=config.speech_update_interval,
            end_of_turn_delay=config.end_of_turn_delay,
            cancel_generation=session.cancel_current_reply,
            finalize_turn=session.finalize_turn,
            report_latency=config.report_latency,
        )
        application.run()
    finally:
        if application is None or application.can_close_resources:
            model.close()


def _run_realtime(config: AppConfig) -> None:
    api_key = os.environ.get(config.realtime_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"--backend realtime requires an API key in the "
            f"{config.realtime_api_key_env} environment variable; do not put "
            "API keys in source code or command-line arguments"
        )

    from .realtime import RealtimeVoiceApplication, build_instructions

    application = RealtimeVoiceApplication(
        api_key=api_key,
        model=config.realtime_model,
        voice=config.realtime_voice,
        instructions=build_instructions(config.system_prompt),
        story_reader=lambda: read_goodnight_story(config.files_dir),
        input_device=config.input_device,
        output_device=config.output_device,
    )
    application.run()
