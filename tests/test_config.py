from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from offline_voice_chat.config import parse_args


class ConfigTests(unittest.TestCase):
    def test_parses_defaults_and_resolves_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()

            config = parse_args(["--model", str(model_path)])

        # A direct run must not touch audio routing, the GPU, or saved memory.
        self.assertEqual(config.model_path, model_path.resolve())
        self.assertEqual(config.backend, "local")
        self.assertIsNone(config.input_device)
        self.assertIsNone(config.output_device)
        self.assertEqual(config.gpu_layers, 0)
        self.assertTrue(config.persistent_memory)
        self.assertFalse(config.clear_memory)
        self.assertFalse(config.select_audio_devices)
        self.assertFalse(config.echo_cancel)

    def test_realtime_backend_does_not_require_a_local_model(self) -> None:
        config = parse_args(["--backend", "realtime"])

        self.assertEqual(config.backend, "realtime")
        self.assertIsNone(config.model_path)

    def test_rejects_invalid_numeric_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()

            invalid_options = (
                ("--context-size", "0"),
                ("--max-tokens", "0"),
                ("--temperature", "-0.1"),
                ("--threads", "0"),
                ("--max-history-turns", "0"),
                ("--speech-update-interval", "0"),
                ("--speech-update-interval", "nan"),
                ("--speech-update-interval", "inf"),
                ("--end-of-turn-delay", "-0.1"),
                ("--end-of-turn-delay", "nan"),
                ("--end-of-turn-delay", "11"),
                ("--max-tokens", "4096"),
            )

            for option, value in invalid_options:
                with (
                    self.subTest(option=option),
                    self.assertRaises(SystemExit),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    parse_args(["--model", str(model_path), option, value])


if __name__ == "__main__":
    unittest.main()
