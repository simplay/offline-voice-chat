from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from offline_voice_chat.memory import (
    PersistentConversationMemory,
)


class PersistentConversationMemoryTests(unittest.TestCase):
    def test_creates_and_restores_a_compact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = PersistentConversationMemory(path)
            self.assertFalse(memory.load())

            saved = memory.record_complete_turn(
                "My name is Ana. I prefer tea without sugar.",
                "I'll remember that preference for later.",
            )
            restored = PersistentConversationMemory(path)

            self.assertTrue(saved)
            self.assertTrue(restored.load())
            context = restored.context_for_chat()

        self.assertIsNotNone(context)
        self.assertIn("User: My name is Ana.", context)
        self.assertIn("User: I prefer tea without sugar.", context)
        self.assertIn("Assistant: I'll remember", context)

    def test_corrupt_or_incompatible_storage_is_ignored_not_overwritten(
        self,
    ) -> None:
        documents = ("not json", json.dumps({"schema_version": 99, "entries": []}))
        for original in documents:
            with (
                self.subTest(original=original),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "memory.json"
                path.write_text(original, encoding="utf-8")
                warnings: list[str] = []
                memory = PersistentConversationMemory(path, on_warning=warnings.append)

                self.assertFalse(memory.load())
                self.assertIsNone(memory.context_for_chat())
                self.assertFalse(
                    memory.record_complete_turn(
                        "I prefer a safe fallback.", "I will use it."
                    )
                )
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                self.assertEqual(len(warnings), 1)

    def test_clear_removes_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory = PersistentConversationMemory(path)
            memory.record_complete_turn("My name is Bea.", "I will remember your name.")

            self.assertTrue(memory.clear())

            self.assertFalse(path.exists())
            self.assertIsNone(memory.context_for_chat())


if __name__ == "__main__":
    unittest.main()
