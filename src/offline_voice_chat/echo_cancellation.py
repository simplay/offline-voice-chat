"""Echo cancellation for this app through PulseAudio's WebRTC module."""

from __future__ import annotations

import os
import shlex
import signal
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .pulse_audio import default_pulse_devices, query_pulse_audio, run_pactl


@contextmanager
def echo_cancelled_devices(
    input_device: int | str | None,
    output_device: int | str | None,
) -> Iterator[tuple[int, int]]:
    """
    Yields the PortAudio indices to use. The module is removed again on exit,
    also after an error or SIGTERM.
    """

    if sys.platform != "linux":
        raise RuntimeError("--echo-cancel requires Linux with PipeWire or PulseAudio")

    indices = _pulse_route_indices(input_device, output_device)
    source, sink = _desktop_microphone_and_speaker()

    name = f"offline_voice_chat_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    source_name = f"{name}_source"
    sink_name = f"{name}_sink"
    module_id = None
    previous_environment = {
        key: os.environ.get(key) for key in ("PULSE_SOURCE", "PULSE_SINK")
    }
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_session)

    try:
        module_id = _load_echo_cancel_module(source, sink, source_name, sink_name)
        _wait_for_devices(source_name, sink_name)

        # ALSA's pulse plugin reads these when opening the session's streams.
        # Both directions must use this pair for playback to become the AEC reference.
        os.environ["PULSE_SOURCE"] = source_name
        os.environ["PULSE_SINK"] = sink_name
        print(
            "Acoustic echo cancellation: WebRTC enabled\n"
            f"  Microphone: {source.get('description', source['name'])}\n"
            f"  Speakers: {sink.get('description', sink['name'])}",
            flush=True,
        )
        yield indices

    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        try:
            _unload_session_module(module_id, source_name)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


def _pulse_route_indices(
    input_device: int | str | None,
    output_device: int | str | None,
) -> tuple[int, int]:
    try:
        import sounddevice
    except (ImportError, OSError) as exception:
        raise RuntimeError(
            f"could not load sounddevice/PortAudio: {exception}"
        ) from exception

    indices = []
    for direction, requested in (("input", input_device), ("output", output_device)):
        device_name = "pulse"

        if requested is not None:
            device_name = requested

        try:
            device = sounddevice.query_devices(device_name, direction)
        except (sounddevice.PortAudioError, ValueError, OSError) as exception:
            raise RuntimeError(
                f"echo cancellation cannot use the {direction} device: {exception}"
            ) from exception

        if device["name"] != "pulse":
            raise RuntimeError(
                f"echo cancellation requires the desktop pulse route for {direction}, "
                f"but {device['name']!r} is selected; choose pulse in Configure audio "
                "and select the physical device in Linux sound settings, or use "
                "--no-echo-cancel"
            )

        indices.append(device["index"])

    return indices[0], indices[1]


def _desktop_microphone_and_speaker() -> tuple[dict[str, Any], dict[str, Any]]:
    defaults = default_pulse_devices()
    if "input" not in defaults or "output" not in defaults:
        raise RuntimeError(
            "echo cancellation needs a desktop microphone and speaker; check Linux sound settings"
        )

    source = defaults["input"]
    device_class = source.get("properties", {}).get("device.class")
    is_monitor = source["name"].endswith(".monitor") or device_class == "monitor"
    if is_monitor:
        raise RuntimeError(
            "echo cancellation needs a microphone, not a speaker monitor source"
        )

    return source, defaults["output"]


def _load_echo_cancel_module(
    source: dict[str, Any],
    sink: dict[str, Any],
    source_name: str,
    sink_name: str,
) -> str:
    """
    Returns the module ID.
    """

    try:
        result = run_pactl(
            "load-module",
            "module-echo-cancel",
            "aec_method=webrtc",
            f"source_master={shlex.quote(source['name'])}",
            f"sink_master={shlex.quote(sink['name'])}",
            f"source_name={source_name}",
            f"sink_name={sink_name}",
            "source_properties=priority.session=0",
            "sink_properties=priority.session=0",
            "rate=48000",
            "channels=1",
            "channel_map=mono",
        )
    except RuntimeError as exception:
        raise RuntimeError(
            f"could not enable WebRTC echo cancellation: {exception}; the audio server "
            "must provide module-echo-cancel with WebRTC support. "
            "Use --no-echo-cancel to run without it"
        ) from exception

    if not result.isdecimal():
        raise RuntimeError("echo cancellation did not return a valid module ID")

    return result


def _unload_session_module(module_id: str | None, source_name: str) -> None:
    """
    Prints a manual fix instead of raising, so a failed cleanup never hides
    the original error.
    """

    try:
        if module_id is None:
            # A timed-out or interrupted client can lose the reply after
            # the server created the module. Recover only our unique name.
            module_id = _find_session_module(source_name)

        if module_id is not None:
            run_pactl("unload-module", module_id)
    except RuntimeError as exception:
        hint = (
            f"pactl unload-module {module_id}"
            if module_id is not None
            else f"pactl list modules short (look for {source_name})"
        )
        print(
            f"Could not remove echo cancellation: {exception}\n"
            f"Check this session's module with: {hint}",
            file=sys.stderr,
        )


def _find_session_module(source_name: str) -> str | None:
    for module in query_pulse_audio("list", "modules"):
        if module.get("name") != "module-echo-cancel":
            continue

        try:
            arguments = shlex.split(module.get("argument", ""))
        except ValueError:
            continue

        if f"source_name={source_name}" in arguments:
            return str(module["index"])

    return None


def _terminate_session(signum: int, frame: object) -> None:
    # Turn SIGTERM into SystemExit, so echo_cancelled_devices() unloads
    # its module.
    raise SystemExit(128 + signum)


def _wait_for_devices(source_name: str, sink_name: str) -> None:
    deadline = time.monotonic() + 2
    while True:
        source_names = {item["name"] for item in query_pulse_audio("list", "sources")}
        sink_names = {item["name"] for item in query_pulse_audio("list", "sinks")}
        if source_name in source_names and sink_name in sink_names:
            return

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "echo-cancelled microphone and speaker did not become available"
            )

        time.sleep(0.05)
