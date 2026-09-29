from __future__ import annotations

import unittest
from datetime import datetime

from fakes import (
    FakeContextTool,
    FakeMemory,
    FakeStreamingModel,
)

from offline_voice_chat.chat import ChatSession
from offline_voice_chat.commands import (
    ONLINE_COMMAND_PREFIX,
    TERMINATE_PHRASES,
    default_commands,
    parse_prefix_request,
)


class CommandTests(unittest.TestCase):
    def test_online_prefix_keeps_the_query_and_accepts_spoken_punctuation(self) -> None:
        self.assertEqual(
            parse_prefix_request(
                "Please, check online: news in Switzerland!", ONLINE_COMMAND_PREFIX
            ),
            "news in Switzerland",
        )
        self.assertEqual(
            parse_prefix_request("Please check online,news", ONLINE_COMMAND_PREFIX),
            "news",
        )
        self.assertEqual(
            parse_prefix_request("Please check online", ONLINE_COMMAND_PREFIX),
            "",
        )
        self.assertIsNone(
            parse_prefix_request(
                "Do not please check online news", ONLINE_COMMAND_PREFIX
            )
        )

    def test_time_command_skips_model_and_persistent_memory(self) -> None:
        model = FakeStreamingModel([])
        memory = FakeMemory()
        session = ChatSession(
            model, system_prompt="Be brief.", max_history_turns=2, memory=memory
        )
        session.commands = default_commands(
            reset_conversation=session.reset_conversation,
            now=lambda: datetime(2026, 9, 5, 12, 34),
        )
        reply_text = "".join(session.stream_reply("What time is it?"))
        self.assertEqual(reply_text, "It is 12:34.")
        session.finalize_turn(True)
        self.assertEqual(model.calls, [])
        self.assertEqual(memory.completed_turns, [])

    def test_commands_require_a_whole_explicit_phrase(self) -> None:
        registry = default_commands(reset_conversation=lambda: "Cleared.")
        self.assertIsNone(registry.match("Do not clear conversation."))
        self.assertIsNone(registry.match("Tell me why people ask what time is it."))
        self.assertEqual(registry.match("CLEAR conversation!").run(), "Cleared.")
        self.assertIn(TERMINATE_PHRASES[0], registry.help_text())

    def test_reset_clears_recent_history_and_file_selection(self) -> None:
        tool = FakeContextTool(None)
        session = ChatSession(
            FakeStreamingModel([["An earlier answer."]]),
            system_prompt="Be brief.",
            max_history_turns=2,
            context_tool=tool,
        )
        session.commands = default_commands(
            reset_conversation=session.reset_conversation
        )
        list(session.stream_reply("Earlier question"))
        list(session.stream_reply("clear conversation"))
        self.assertEqual(tool.reset_count, 1)
        self.assertEqual(len(session.history), 2)
        self.assertEqual(session.history[0]["content"], "clear conversation")


if __name__ == "__main__":
    unittest.main()
