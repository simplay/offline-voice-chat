from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import unittest
from unittest import mock

from fakes import FakePortAudioError

from offline_voice_chat.echo_cancellation import echo_cancelled_devices


class PulseRouteSounddevice:
    """Stand-in for sounddevice where every device query finds one device."""

    PortAudioError = FakePortAudioError

    def __init__(self, device: dict[str, object]) -> None:
        self.query_devices = mock.Mock(return_value=device)


class EchoCancellationTests(unittest.TestCase):
    def setUp(self):
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch("sys.platform", "linux"))
        self.patches.enter_context(mock.patch.dict(os.environ))
        os.environ.pop("PULSE_SOURCE", None)
        os.environ.pop("PULSE_SINK", None)
        self.patches.enter_context(
            mock.patch(
                "offline_voice_chat.pulse_audio.shutil.which",
                return_value="/usr/bin/pactl",
            )
        )
        pulse_route = {"name": "pulse", "index": 14}
        self.portaudio = PulseRouteSounddevice(pulse_route)
        self.patches.enter_context(
            mock.patch.dict("sys.modules", {"sounddevice": self.portaudio})
        )
        self.sources = [{"name": "system-mic", "description": "SoundWire microphones"}]
        self.sinks = [{"name": "system-speaker", "description": "Built-in speaker"}]
        self.commands = []
        self.loaded = None
        self.load_error = None
        self.load_reply = "42\n"
        self.load_reply_error = None
        self.unload_error = None
        self.devices_ready = True
        self.output = io.StringIO()
        self.errors = io.StringIO()
        self.patches.enter_context(contextlib.redirect_stdout(self.output))
        self.patches.enter_context(contextlib.redirect_stderr(self.errors))
        self.run = self.patches.enter_context(
            mock.patch(
                "offline_voice_chat.pulse_audio.subprocess.run", side_effect=self.pactl
            )
        )

    def pactl(self, arguments, **kwargs):
        self.assertEqual(arguments[0], "/usr/bin/pactl")
        self.assertEqual(kwargs["timeout"], 2)
        self.assertTrue(kwargs["check"])
        self.assertNotIn("shell", kwargs)
        command = arguments[1:]
        self.commands.append(command)

        if command[0] == "load-module":
            if self.load_error:
                raise self.load_error

            self.assertEqual(command[1], "module-echo-cancel")
            module_arguments = [item.split("=", 1) for item in command[2:]]
            self.loaded = dict(module_arguments)
            if self.load_reply_error:
                raise self.load_reply_error

            return subprocess.CompletedProcess(arguments, 0, stdout=self.load_reply)

        if command[0] == "unload-module":
            self.assertEqual(command, ["unload-module", "42"])
            if self.unload_error:
                raise self.unload_error

            self.loaded = None
            return subprocess.CompletedProcess(arguments, 0, stdout="")

        self.assertEqual(command[0], "--format=json")

        query = command[1:]
        if query == ["info"]:
            reply = {
                "default_source_name": "system-mic",
                "default_sink_name": "system-speaker",
            }
        elif query == ["list", "modules"]:
            reply = self.module_list()
        elif query == ["list", "sources"]:
            reply = self.device_list(self.sources, "source_name")
        elif query == ["list", "sinks"]:
            reply = self.device_list(self.sinks, "sink_name")
        else:
            self.fail(f"Unexpected server command: {command}")

        return subprocess.CompletedProcess(arguments, 0, stdout=json.dumps(reply))

    def module_list(self) -> list[dict[str, object]]:
        """List another program's echo module, plus ours while it is loaded."""

        modules = [
            {
                "index": 41,
                "name": "module-echo-cancel",
                "argument": "source_name=someone_else",
            }
        ]

        if self.loaded:
            argument = " ".join(f"{key}={text}" for key, text in self.loaded.items())
            modules.append(
                {"index": 42, "name": "module-echo-cancel", "argument": argument}
            )

        return modules

    def device_list(
        self, devices: list[dict[str, str]], loaded_name_key: str
    ) -> list[dict[str, str]]:
        """List desktop devices, plus our virtual device once it is ready."""

        listed = list(devices)

        if self.loaded and self.devices_ready:
            listed.append({"name": self.loaded[loaded_name_key]})

        return listed

    def test_routes_both_streams_through_webrtc_and_removes_only_its_module(self):
        handler = signal.getsignal(signal.SIGTERM)
        with echo_cancelled_devices(14, 14) as indices:
            self.assertEqual(indices, (14, 14))
            self.assertEqual(self.loaded["aec_method"], "webrtc")
            self.assertEqual(self.loaded["source_master"], "system-mic")
            self.assertEqual(self.loaded["sink_master"], "system-speaker")
            self.assertEqual(os.environ["PULSE_SOURCE"], self.loaded["source_name"])
            self.assertEqual(os.environ["PULSE_SINK"], self.loaded["sink_name"])
            self.assertNotEqual(os.environ["PULSE_SOURCE"], os.environ["PULSE_SINK"])

        self.assertIsNone(self.loaded)
        self.assertNotIn("PULSE_SOURCE", os.environ)
        self.assertNotIn("PULSE_SINK", os.environ)
        self.assertEqual(signal.getsignal(signal.SIGTERM), handler)
        self.assertIn("WebRTC enabled", self.output.getvalue())

    def test_backend_errors_and_interruptions_still_remove_the_module(self):
        for failure in (
            RuntimeError("backend failed"),
            KeyboardInterrupt(),
            SystemExit(143),
        ):
            with self.subTest(failure=failure):
                with self.assertRaises(type(failure)):
                    with echo_cancelled_devices(None, None):
                        raise failure

                self.assertIsNone(self.loaded)
                self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_missing_pactl_and_unsupported_platform_have_actionable_errors(self):
        with mock.patch(
            "offline_voice_chat.pulse_audio.shutil.which", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "apt install pulseaudio-utils"):
                with echo_cancelled_devices(None, None):
                    self.fail("Must not enter the backend")

        with mock.patch("sys.platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "requires Linux"):
                with echo_cancelled_devices(None, None):
                    self.fail("Must not enter the backend")

        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
