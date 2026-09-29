from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from offline_voice_chat.config import parse_args
from offline_voice_chat.startup_menu import configure_startup


class StartupMenuTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # The app resolves paths, and macOS's temp folder is a symlink.
        self.directory = Path(directory.name).resolve()
        (self.directory / "default.txt").write_text("Be brief.\n", encoding="utf-8")
        self.config = parse_args(
            [
                "--backend",
                "realtime",
                "--system-prompts-dir",
                str(self.directory),
                "--input-device",
                "7",
                "--output-device",
                "23",
            ]
        )
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.output = io.StringIO()
        self.patches.enter_context(contextlib.redirect_stdout(self.output))
        self.patches.enter_context(mock.patch("sys.stdin.isatty", return_value=True))
        self.enter = self.patches.enter_context(mock.patch("builtins.input"))
        self.audio = self.patches.enter_context(
            mock.patch("offline_voice_chat.startup_menu.select_audio_devices")
        )

    def test_start_keeps_manually_selected_settings(self):
        for answer in ("", "1"):
            with self.subTest(answer=answer):
                self.enter.side_effect = [answer]
                self.assertIs(configure_startup(self.config), self.config)

        self.audio.assert_not_called()
        self.assertIn("System prompt: default.txt", self.output.getvalue())

    def test_prompt_selection_lists_files_and_replaces_loaded_text(self):
        custom = self.directory / "German.txt"
        custom.write_text("Antworte höflich auf Deutsch.\n", encoding="utf-8")
        (self.directory / "notes.md").write_text("Not a prompt", encoding="utf-8")
        self.enter.side_effect = ["invalid", "3", "invalid", "0", "9", "2", "1"]
        result = configure_startup(self.config)
        self.assertEqual(result.system_prompt, "Antworte höflich auf Deutsch.")
        self.assertEqual(result.system_prompt_file, custom)
        self.assertIn("1. default.txt [loaded]", self.output.getvalue())
        self.assertIn("2. German.txt", self.output.getvalue())
        self.assertIn("System prompt: German.txt", self.output.getvalue())

    def test_clear_saved_memory_requires_confirmation(self):
        memory_path = self.directory / "saved-memory.json"
        memory_path.write_text("keep this memory", encoding="utf-8")
        config = replace(self.config, memory_file=memory_path)
        self.enter.side_effect = ["4", "", "q"]

        self.assertIsNone(configure_startup(config))

        self.assertEqual(memory_path.read_text(encoding="utf-8"), "keep this memory")
        self.assertIn("Saved memory kept.", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
