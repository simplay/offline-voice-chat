"""Interfaces that ChatSession depends on; tests pass fakes for them."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Protocol

ChatMessage = dict[str, str]


class StreamingChatModel(Protocol):
    prompt_budget: int

    def count_tokens(self, text: str) -> int:
        """
        Counted with the model's tokenizer.
        """

    def prepare_reply(self) -> None:
        """
        Called before each reply to clear an earlier cancellation.
        """

    def cancel_current_reply(self) -> None:
        """
        May be called from another thread.
        """

    def stream_reply(
        self, messages: Sequence[ChatMessage]
    ) -> Generator[str, None, None]:
        """
        The caller may close the generator early.
        """


class RequestContextTool(Protocol):
    """
    Adds file text to a request when the user asks for a file.
    """

    def context_for_request(self, request: str) -> str | None:
        """
        Returns None when the request needs no extra text.
        """

    def reset(self) -> None:
        """
        Forgets the file chosen in earlier turns.
        """


class ConversationMemory(Protocol):
    """
    Memory that survives restarts.
    """

    def context_for_chat(self) -> str | None:
        """
        Returns None when nothing is stored.
        """

    def record_complete_turn(self, user_text: str, assistant_text: str) -> bool:
        """
        Must not raise, so a storage error never breaks a turn.
        """
