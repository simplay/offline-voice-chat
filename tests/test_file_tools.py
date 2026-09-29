from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from offline_voice_chat.file_tools import LocalTextFileReader, read_goodnight_story


class LocalTextFileReaderTests(unittest.TestCase):
    def test_story_command_reads_only_the_named_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "goodnight_story.txt").write_text(
                "The hedgehog slept well.\n", encoding="utf-8"
            )
            (root / "other.txt").write_text("Not the story.", encoding="utf-8")

            self.assertEqual(read_goodnight_story(root), "The hedgehog slept well.")
            (root / "goodnight_story.txt").unlink()
            self.assertIn("could not open", read_goodnight_story(root))

    def test_reads_named_utf8_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "deer.txt").write_text(
                "Deer rest beneath the summer leaves.",
                encoding="utf-8",
            )
            reader = LocalTextFileReader(root)

            for request in ("Please read deer dot txt", "Read deer dot t, x t"):
                with self.subTest(request=request):
                    context = reader.context_for_request(request)
                    self.assertIn("File: deer.txt", context)
                    self.assertIn(
                        "BEGIN FILE ---\nDeer rest beneath the summer leaves.\n"
                        "--- END FILE",
                        context,
                    )

    def test_reports_missing_and_oversized_files_without_inventing_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "large.txt").write_text("too large", encoding="utf-8")
            (root / "binary.txt").write_bytes(b"\xff")
            reader = LocalTextFileReader(root, max_bytes=3)

            missing = reader.context_for_request("Read missing.txt")
            oversized = reader.context_for_request("Read large.txt")
            invalid = reader.context_for_request("Read binary.txt")

        self.assertIn("file does not exist", missing)
        self.assertIn("exceeds the 3-byte read limit", oversized)
        self.assertIn("not valid UTF-8 text", invalid)

    def test_links_leaving_the_files_directory_are_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "files"
            root.mkdir()
            secret = Path(directory) / "secret.txt"
            secret.write_text("Private notes.", encoding="utf-8")

            try:
                (root / "notes.txt").symlink_to(secret)
                (root / "goodnight_story.txt").symlink_to(secret)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            file_context = LocalTextFileReader(root).context_for_request(
                "Read notes.txt"
            )
            story = read_goodnight_story(root)

        self.assertIn("outside the allowed directory", file_context)
        self.assertNotIn("Private notes.", file_context)
        self.assertIn("outside the allowed directory", story)
        self.assertNotIn("Private notes.", story)


if __name__ == "__main__":
    unittest.main()
