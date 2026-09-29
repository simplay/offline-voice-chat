from __future__ import annotations

import unittest

from fakes import (
    FakeContextTool,
    FakeLlama,
    FakeLlamaCpp,
    FakeMemory,
    FakeStreamingModel,
    load_llama_model,
    text_chunk,
)

from offline_voice_chat.chat import ChatSession


class ChatSessionTests(unittest.TestCase):
    def test_failed_delivery_restores_history_evicted_by_the_provisional_turn(
        self,
    ) -> None:
        memory = FakeMemory()
        session = ChatSession(
            FakeStreamingModel([["Heard answer"], ["Unheard answer"]]),
            system_prompt="Be brief.",
            max_history_turns=1,
            memory=memory,
        )
        list(session.stream_reply("Heard question"))
        session.finalize_turn(True)
        list(session.stream_reply("Unheard question"))
        session.finalize_turn(False)
        self.assertEqual(session.history[0]["content"], "Heard question")
        self.assertEqual(len(session.history), 2)
        self.assertEqual(memory.completed_turns, [("Heard question", "Heard answer")])

    def test_streams_reply_and_records_complete_turn(self) -> None:
        model = FakeStreamingModel([["Hello", " there"]])
        session = ChatSession(
            model,
            system_prompt="Be concise.",
            max_history_turns=8,
        )

        response = "".join(session.stream_reply("Hi"))

        self.assertEqual(response, "Hello there")
        self.assertEqual(
            model.calls[0],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        )
        self.assertEqual(
            session.history,
            (
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there"},
            ),
        )

    def test_trims_oldest_complete_turns(self) -> None:
        model = FakeStreamingModel([["one"], ["two"], ["three"]])
        session = ChatSession(
            model,
            system_prompt="Be concise.",
            max_history_turns=2,
        )

        for transcript in ("first", "second", "third"):
            list(session.stream_reply(transcript))

        self.assertEqual(
            session.history,
            (
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "third"},
                {"role": "assistant", "content": "three"},
            ),
        )

    def test_preserves_partial_answer_when_reply_is_closed(self) -> None:
        model = FakeStreamingModel([["Partial", " remainder"]])
        session = ChatSession(
            model,
            system_prompt="Be concise.",
            max_history_turns=8,
        )

        reply = session.stream_reply("Hi")
        self.assertEqual(next(reply), "Partial")
        reply.close()

        self.assertTrue(model.closed)
        self.assertEqual(
            session.history,
            (
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Partial"},
            ),
        )

    def test_adds_file_tool_context_without_storing_it_in_history(self) -> None:
        model = FakeStreamingModel([["The file describes deer."]])
        tool = FakeContextTool("[FILE DATA]\nDeer in a summer forest.")
        session = ChatSession(
            model,
            system_prompt="Be concise.",
            max_history_turns=8,
            context_tool=tool,
        )

        response = "".join(session.stream_reply("Read deer.txt"))

        self.assertEqual(response, "The file describes deer.")
        self.assertEqual(tool.calls, ["Read deer.txt"])
        self.assertIn("[FILE DATA]", model.calls[0][-1]["content"])
        self.assertEqual(
            session.history,
            (
                {"role": "user", "content": "Read deer.txt"},
                {
                    "role": "assistant",
                    "content": "The file describes deer.",
                },
            ),
        )

    def test_persists_only_a_fully_completed_turn(self) -> None:
        memory = FakeMemory()
        session = ChatSession(
            FakeStreamingModel([["Complete answer."]]),
            system_prompt="Be concise.",
            max_history_turns=8,
            memory=memory,
        )

        list(session.stream_reply("Remember this complete request."))

        self.assertEqual(memory.completed_turns, [])
        session.finalize_turn(True)

        self.assertEqual(
            memory.completed_turns,
            [("Remember this complete request.", "Complete answer.")],
        )


class LlamaCppChatModelTests(unittest.TestCase):
    def test_token_limit_does_not_report_a_truncated_answer_as_complete(self) -> None:
        llama = FakeLlama([text_chunk("Partial"), text_chunk("", "length")])
        model = load_llama_model(FakeLlamaCpp(llama))
        reply = model.stream_reply([{"role": "user", "content": "Hi"}])
        self.assertEqual(next(reply), "Partial")

        with self.assertRaisesRegex(RuntimeError, "before completion"):
            list(reply)

    def test_extracts_only_text_content_from_stream(self) -> None:
        role_chunk = {
            "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]
        }
        llama = FakeLlama(
            [role_chunk, text_chunk("Hello"), text_chunk("!"), text_chunk("", "stop")]
        )
        model = load_llama_model(FakeLlamaCpp(llama))
        messages = [{"role": "user", "content": "Hi"}]

        self.assertEqual(list(model.stream_reply(messages)), ["Hello", "!"])
        self.assertEqual(
            llama.calls[0],
            {
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.2,
                "stream": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
