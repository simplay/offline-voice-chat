"""Choose PortAudio devices before loading models or opening audio streams."""

from __future__ import annotations

import sys
from typing import Any

from .pulse_audio import default_pulse_devices


def select_audio_devices(
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    *,
    interactive: bool = True,
    cancel_message: str = "Startup cancelled.",
) -> tuple[int, int] | None:
    """
    Return the chosen (input, output) PortAudio indices, or None on cancel.
    """

    if interactive and not sys.stdin.isatty():
        raise RuntimeError(
            "audio selection requires an interactive terminal; "
            "run the Linux launcher in a terminal"
        )

    try:
        import sounddevice
    except (ImportError, OSError) as exception:
        raise RuntimeError(
            f"could not load sounddevice/PortAudio: {exception}"
        ) from exception

    try:
        devices = list(sounddevice.query_devices())
        host_apis = sounddevice.query_hostapis()
    except (sounddevice.PortAudioError, OSError) as exception:
        raise RuntimeError(f"could not list audio devices: {exception}") from exception

    input_devices = [device for device in devices if device["max_input_channels"] > 0]
    output_devices = [device for device in devices if device["max_output_channels"] > 0]

    if not input_devices:
        raise RuntimeError(
            "no microphone devices are available; connect a device and check "
            "your Linux sound settings"
        )

    if not output_devices:
        raise RuntimeError(
            "no audio output devices are available; connect a device and check "
            "your Linux sound settings"
        )

    desktop_names = _desktop_device_descriptions(devices)
    input_name = desktop_names.get("input")
    output_name = desktop_names.get("output")

    if interactive:
        print("Choose your audio devices. Enter accepts the default; q cancels.")

    try:
        microphone = _choose_device(
            sounddevice,
            "input",
            "Microphone",
            input_devices,
            requested=input_device,
            host_apis=host_apis,
            desktop_name=input_name,
            interactive=interactive,
        )
        if microphone is None:
            print(cancel_message)
            return None

        speaker = _choose_device(
            sounddevice,
            "output",
            "Speakers/headphones",
            output_devices,
            requested=output_device,
            host_apis=host_apis,
            desktop_name=output_name,
            interactive=interactive,
        )
        if speaker is None:
            print(cancel_message)
            return None
    except (KeyboardInterrupt, EOFError):
        print(f"\n{cancel_message}")
        return None

    print(f"\nMicrophone: {_device_label(microphone, input_name)}")
    print(f"Audio output: {_device_label(speaker, output_name)}\n")
    return microphone["index"], speaker["index"]


def _choose_device(
    sounddevice: Any,
    direction: str,
    label: str,
    devices: list[dict[str, Any]],
    *,
    requested: int | str | None,
    host_apis: Any,
    desktop_name: str | None,
    interactive: bool,
) -> dict[str, Any] | None:
    suggested = requested
    pulse_indices = [device["index"] for device in devices if device["name"] == "pulse"]

    if requested is None and desktop_name and pulse_indices:
        # Prefer the pulse route: the desktop audio server applies the user's
        # device choice and channel setup, which raw ALSA devices bypass.
        suggested = pulse_indices[0]

    try:
        default_index = sounddevice.query_devices(suggested, direction)["index"]
    except (sounddevice.PortAudioError, ValueError, OSError):
        default_index = None

        if requested is not None:
            print(f"Requested {direction} device {requested!r} is unavailable.")

    if interactive:
        return _prompt_device(label, devices, host_apis, default_index, desktop_name)

    for device in devices:
        if device["index"] == default_index:
            return device

    raise RuntimeError(
        f"no usable default {direction} device; use Configure audio to choose a device"
    )


def _desktop_device_descriptions(devices: list[dict[str, Any]]) -> dict[str, str]:
    has_pulse_route = any(device["name"] == "pulse" for device in devices)
    if sys.platform != "linux" or not has_pulse_route:
        return {}

    try:
        pulse_devices = default_pulse_devices()
    except RuntimeError:
        # Device selection still works on systems without a reachable audio
        # server or with an older pactl that cannot return JSON.
        return {}

    return {
        direction: device["description"]
        for direction, device in pulse_devices.items()
        if device.get("description")
    }


def _device_label(device: dict[str, Any], desktop_description: str | None) -> str:
    if device["name"] == "pulse" and desktop_description:
        return f"{desktop_description} (via pulse)"

    return device["name"]


def _prompt_device(
    label: str,
    devices: list[dict[str, Any]],
    host_apis: Any,
    default_index: int | None,
    desktop_description: str | None = None,
) -> dict[str, Any] | None:
    print(f"\n{label}:")
    default_number = None

    for number, device in enumerate(devices, start=1):
        marker = ""

        if device["index"] == default_index:
            default_number = number
            marker = " [default]"

        host_api = host_apis[device["hostapi"]]["name"]
        print(
            f"  {number}. {_device_label(device, desktop_description)} "
            f"({host_api}, device {device['index']}){marker}"
        )

    if default_number is not None:
        default_device = devices[default_number - 1]
        print(
            f"Press Enter to keep {_device_label(default_device, desktop_description)}."
        )

    default_hint = f" [Enter: {default_number}]" if default_number is not None else ""
    while True:
        response = input(
            f"Select {label.lower()} (1-{len(devices)}){default_hint}: "
        ).strip()
        if response.lower() == "q":
            return None

        if not response and default_number is not None:
            return devices[default_number - 1]

        number = int(response) if response.isdigit() else 0

        if 1 <= number <= len(devices):
            return devices[number - 1]

        print(f"Enter a number from 1 to {len(devices)}, or q to cancel.")
