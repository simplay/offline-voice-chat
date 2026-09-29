from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from offline_voice_chat.config import parse_args


class SystemPromptTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # The app resolves paths, and macOS's temp folder is a symlink.
        self.directory = Path(directory.name).resolve()
        self.default = self.directory / "default.txt"
        self.default.write_text("Keep answers short.\n", encoding="utf-8")
        self.arguments = [
            "--backend",
            "realtime",
            "--system-prompts-dir",
            str(self.directory),
        ]

    def test_default_file_is_read_again_at_each_startup(self):
        self.assertEqual(
            parse_args(self.arguments).system_prompt, "Keep answers short."
        )
        self.default.write_text("Respond in German.\n", encoding="utf-8")
        config = parse_args(self.arguments)
        self.assertEqual(config.system_prompt, "Respond in German.")
        self.assertEqual(config.system_prompt_file, self.default)

    def test_custom_file_and_text_override_the_default(self):
        custom = self.directory / "custom prompt.txt"
        custom.write_text("Give clear examples.\n", encoding="utf-8")
        file_config = parse_args([*self.arguments, "--system-prompt-file", str(custom)])
        self.assertEqual(file_config.system_prompt, "Give clear examples.")
        self.assertEqual(file_config.system_prompt_file, custom)
        text_config = parse_args([*self.arguments, "--system-prompt", "  Be brief.  "])
        self.assertEqual(text_config.system_prompt, "Be brief.")
        self.assertIsNone(text_config.system_prompt_file)

    def test_missing_default_stops_startup_with_its_path(self):
        arguments = [
            "--backend",
            "realtime",
            "--system-prompts-dir",
            str(self.directory / "missing"),
        ]
        errors = io.StringIO()

        with self.assertRaises(SystemExit), contextlib.redirect_stderr(errors):
            parse_args(arguments)

        self.assertIn(str(self.directory / "missing/default.txt"), errors.getvalue())


if __name__ == "__main__":
    unittest.main()
