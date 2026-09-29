from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from offline_voice_chat.cli import _run_local, main
from offline_voice_chat.config import AppConfig
from offline_voice_chat.memory import PersistentConversationMemory


class CliTests(unittest.TestCase):
    def saved_memory(self, directory: str) -> Path:
        path = Path(directory) / "memory.json"
        PersistentConversationMemory(path).record_complete_turn(
            "My name is Eli.", "I will remember your name."
        )
        return path

    def test_online_and_story_commands_reach_the_local_voice_session(self) -> None:
        model = mock.Mock(gpu_offload_available=True, prompt_budget=3584)
        application = mock.Mock(can_close_resources=True)
        answers = []

        def run_commands() -> None:
            stream_reply = voice.call_args.args[0]
            online_answer = "".join(stream_reply("Please check online news"))
            story_answer = "".join(stream_reply("Read the goodnight story"))
            answers.extend([online_answer, story_answer])

        with (
            mock.patch("offline_voice_chat.cli.LlamaCppChatModel", return_value=model),
            mock.patch("offline_voice_chat.cli.OnlineLookup") as online,
            mock.patch(
                "offline_voice_chat.cli.read_goodnight_story",
                return_value="Goodnight, little hedgehog.",
            ) as story,
            mock.patch(
                "offline_voice_chat.cli.VoiceChatApplication",
                return_value=application,
            ) as voice,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            online.return_value.check.return_value = "BBC World headlines: One."
            application.run.side_effect = run_commands

            _run_local(AppConfig(Path("model.gguf")), None)

        self.assertEqual(
            answers, ["BBC World headlines: One.", "Goodnight, little hedgehog."]
        )
        online.return_value.check.assert_called_once_with("news")
        story.assert_called_once_with(None)
        model.stream_reply.assert_not_called()
        model.close.assert_called_once()

    def test_echo_cancellation_surrounds_both_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()

            for backend, failure, code in (
                ("local", None, 0),
                ("realtime", None, 0),
                ("realtime", RuntimeError("backend failed"), 1),
            ):
                with self.subTest(backend=backend, failure=failure):
                    events = []

                    @contextlib.contextmanager
                    def echo_cancel(input_device, output_device):
                        self.assertEqual((input_device, output_device), (7, 23))
                        events.append("enable echo cancellation")

                        try:
                            yield 14, 14
                        finally:
                            events.append("remove echo cancellation")

                    def run_backend(config, *args):
                        self.assertEqual(
                            (config.input_device, config.output_device), (14, 14)
                        )
                        events.append("backend")

                        if failure is not None:
                            raise failure

                    with (
                        mock.patch(
                            "offline_voice_chat.cli.echo_cancelled_devices",
                            side_effect=echo_cancel,
                        ),
                        mock.patch(
                            f"offline_voice_chat.cli._run_{backend}",
                            side_effect=run_backend,
                        ),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        self.assertEqual(
                            main(
                                [
                                    "--backend",
                                    backend,
                                    "--model",
                                    str(model),
                                    "--no-persistent-memory",
                                    "--echo-cancel",
                                    "--input-device",
                                    "7",
                                    "--output-device",
                                    "23",
                                ]
                            ),
                            code,
                        )

                    self.assertEqual(
                        events,
                        [
                            "enable echo cancellation",
                            "backend",
                            "remove echo cancellation",
                        ],
                    )

    def test_realtime_backend_fails_cleanly_without_api_key(self) -> None:
        error_output = io.StringIO()

        with (
            mock.patch.dict(os.environ, {"TEST_OPENAI_KEY": ""}),
            contextlib.redirect_stderr(error_output),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = main(
                [
                    "--backend",
                    "realtime",
                    "--realtime-api-key-env",
                    "TEST_OPENAI_KEY",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("TEST_OPENAI_KEY environment variable", error_output.getvalue())

    def test_clear_memory_removes_custom_store_before_local_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            memory_path = self.saved_memory(directory)

            with (
                mock.patch("offline_voice_chat.cli._run_local") as run_local,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exit_code = main(
                    [
                        "--model",
                        str(model_path),
                        "--memory-file",
                        str(memory_path),
                        "--clear-memory",
                    ]
                )

            prepared_memory = run_local.call_args.args[1]
            self.assertFalse(memory_path.exists())

        self.assertEqual(exit_code, 0)
        self.assertIsNone(prepared_memory.context_for_chat())


if __name__ == "__main__":
    unittest.main()
