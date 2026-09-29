from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from fakes import FakePortAudioError

from offline_voice_chat.audio_devices import select_audio_devices


class FakeSounddeviceModule:
    """Stand-in for sounddevice whose device queries a test can inspect."""

    PortAudioError = FakePortAudioError

    def __init__(self, query_devices, host_apis: list[dict[str, str]]) -> None:
        self.query_devices = mock.Mock(side_effect=query_devices)
        self.query_hostapis = mock.Mock(return_value=host_apis)


class AudioDeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            dict(
                index=5,
                name="Built-in mic",
                hostapi=0,
                max_input_channels=2,
                max_output_channels=0,
            ),
            dict(
                index=7,
                name="USB microphone",
                hostapi=0,
                max_input_channels=1,
                max_output_channels=0,
            ),
            dict(
                index=9,
                name="Speakers",
                hostapi=0,
                max_input_channels=0,
                max_output_channels=2,
            ),
            dict(
                index=23,
                name="USB headset",
                hostapi=1,
                max_input_channels=1,
                max_output_channels=2,
            ),
        ]
        self.sounddevice = FakeSounddeviceModule(
            self.query_devices, host_apis=[{"name": "ALSA"}, {"name": "Other API"}]
        )
        self.output = io.StringIO()
        self.enter = mock.Mock()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict("sys.modules", {"sounddevice": self.sounddevice}).start()
        mock.patch("sys.stdin.isatty", return_value=True).start()
        mock.patch("builtins.input", self.enter).start()
        self.stdout = contextlib.redirect_stdout(self.output)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def query_devices(self, requested=None, direction=None):
        if direction is None:
            return self.devices

        if requested is None:
            requested = {"input": 5, "output": 9}[direction]

        matches = [
            device
            for device in self.devices
            if requested in (device["index"], device["name"])
            and device[f"max_{direction}_channels"] > 0
        ]
        if len(matches) != 1:
            raise ValueError("Device unavailable")

        return matches[0]

    def test_menus_filter_by_direction_and_return_portaudio_ids(self):
        self.enter.side_effect = ["2", "2"]
        self.assertEqual(select_audio_devices(), (7, 23))
        microphone_menu, playback_menu = self.output.getvalue().split(
            "Speakers/headphones:"
        )
        self.assertIn("USB microphone", microphone_menu)
        self.assertNotIn("Speakers (", microphone_menu)
        self.assertIn("USB headset (Other API, device 23)", playback_menu)
        self.assertNotIn("Built-in mic (", playback_menu)

    def test_automatic_selection_accepts_defaults_and_preserves_overrides(self):
        with mock.patch("sys.stdin.isatty", return_value=False):
            self.assertEqual(select_audio_devices(interactive=False), (5, 9))
            self.assertEqual(select_audio_devices(7, interactive=False), (7, 9))
            self.assertEqual(
                select_audio_devices(output_device=23, interactive=False), (5, 23)
            )

        self.enter.assert_not_called()

    def test_cancelling_either_menu_does_not_choose_devices(self):
        for answers in (["q"], ["1", "Q"], [EOFError], [KeyboardInterrupt]):
            with self.subTest(answers=answers):
                self.enter.side_effect = answers
                self.assertIsNone(select_audio_devices())


if __name__ == "__main__":
    unittest.main()
