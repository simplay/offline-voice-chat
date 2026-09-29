"""Query and control the desktop audio server through its PulseAudio interface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


def run_pactl(*arguments: str) -> str:
    executable = shutil.which("pactl")
    if executable is None:
        raise RuntimeError(
            "pactl is required for desktop echo cancellation; on Ubuntu, install "
            "it with: sudo apt install pulseaudio-utils"
        )

    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except subprocess.CalledProcessError as exception:
        detail = exception.stderr.strip() or str(exception)
        raise RuntimeError(f"pactl {arguments[0]} failed: {detail}") from exception
    except (OSError, subprocess.TimeoutExpired) as exception:
        raise RuntimeError(f"pactl {arguments[0]} failed: {exception}") from exception

    return result.stdout.strip()


def query_pulse_audio(*arguments: str) -> Any:
    try:
        return json.loads(run_pactl("--format=json", *arguments))
    except ValueError as exception:
        raise RuntimeError("pactl returned invalid JSON") from exception


def default_pulse_devices() -> dict[str, dict[str, Any]]:
    """Resolve desktop defaults, honoring this process's PulseAudio overrides."""

    server = query_pulse_audio("info")
    devices = {}

    for direction, kind, field, environment_key in (
        ("input", "sources", "default_source_name", "PULSE_SOURCE"),
        ("output", "sinks", "default_sink_name", "PULSE_SINK"),
    ):
        name = os.environ.get(environment_key) or server.get(field)

        if name in ("@DEFAULT_SOURCE@", "@DEFAULT_SINK@"):
            name = server.get(field)

        if name:
            for device in query_pulse_audio("list", kind):
                if device.get("name") == name:
                    devices[direction] = device
                    break

    return devices
