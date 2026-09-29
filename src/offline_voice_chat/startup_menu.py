"""Configure a voice-chat session before loading models or opening streams."""

from __future__ import annotations

import sys
from dataclasses import replace

from .audio_devices import select_audio_devices
from .config import AppConfig
from .memory import PersistentConversationMemory
from .system_prompts import list_system_prompts, load_system_prompt


def configure_startup(config: AppConfig) -> AppConfig | None:
    if not sys.stdin.isatty():
        raise RuntimeError("the startup menu requires an interactive terminal")

    try:
        while True:
            prompt_name = (
                config.system_prompt_file.name
                if config.system_prompt_file is not None
                else "command-line text"
            )
            print(f"\nSystem prompt: {prompt_name}")
            print(
                f"Acoustic echo cancellation: {'on' if config.echo_cancel else 'off'}"
            )
            print(
                "1) Start\n2) Configure audio\n"
                "3) Load system prompt\n4) Clear saved memory"
            )
            response = (
                input("Choose an option [Enter: Start, q: Quit]: ").strip().lower()
            )

            if response in ("", "1"):
                if config.input_device is None or config.output_device is None:
                    try:
                        selection = select_audio_devices(
                            config.input_device, config.output_device, interactive=False
                        )
                    except RuntimeError as exception:
                        print(f"Could not select default audio: {exception}")
                        continue

                    if selection is None:
                        return None

                    config = config.with_audio_devices(*selection)

                return config

            if response == "q":
                print("Startup cancelled.")
                return None

            if response == "2":
                try:
                    selection = select_audio_devices(
                        config.input_device,
                        config.output_device,
                        cancel_message="Audio configuration cancelled.",
                    )
                except RuntimeError as exception:
                    print(f"Could not configure audio: {exception}")
                    continue

                if selection is not None:
                    config = config.with_audio_devices(*selection)
            elif response == "3":
                config = _select_system_prompt(config)
            elif response == "4":
                _clear_saved_memory(config)
            else:
                print("Choose 1, 2, 3, or 4, or q to quit.")
    except (KeyboardInterrupt, EOFError):
        print("\nStartup cancelled.")
        return None


def _select_system_prompt(config: AppConfig) -> AppConfig:
    try:
        paths = list_system_prompts(config.system_prompts_dir)
    except ValueError as exception:
        print(exception)
        return config

    if not paths:
        print(f"No .txt system prompts found in {config.system_prompts_dir}.")
        return config

    print(f"\nSystem prompts in {config.system_prompts_dir}:")
    for number, path in enumerate(paths, start=1):
        marker = " [loaded]" if path == config.system_prompt_file else ""
        print(f"  {number}. {path.name}{marker}")

    while True:
        response = input("Load prompt number [Enter: Keep current, q: Back]: ").strip()
        if not response or response.lower() == "q":
            return config

        number = int(response) if response.isdigit() else 0

        if not 1 <= number <= len(paths):
            print(f"Choose a number from 1 to {len(paths)}.")
            continue

        path = paths[number - 1]
        try:
            text = load_system_prompt(path)
        except ValueError as exception:
            print(exception)
            continue

        print(f"Loaded system prompt: {path.name}")
        return replace(config, system_prompt=text, system_prompt_file=path)


def _clear_saved_memory(config: AppConfig) -> None:
    response = input(f"Clear saved memory at {config.memory_file}? [y/N]: ")
    response = response.strip().lower()

    if response not in ("y", "yes"):
        print("Saved memory kept.")
        return

    memory = PersistentConversationMemory(
        config.memory_file,
        on_warning=lambda message: print(
            f"Persistent memory warning: {message}", file=sys.stderr, flush=True
        ),
    )

    if memory.clear():
        print(f"Cleared persistent memory: {config.memory_file}")
