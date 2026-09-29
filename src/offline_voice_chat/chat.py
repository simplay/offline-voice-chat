"""
Chat history, spoken commands, and tracking which answers the user heard.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass

from .commands import CommandMatch, CommandRegistry
from .contracts import (
    ChatMessage,
    ConversationMemory,
    RequestContextTool,
    StreamingChatModel,
)
from .prompts import PromptBuilder

# How often a turn that waits for a blocking command checks for cancellation.
_CANCEL_POLL_SECONDS = 0.05


@dataclass
class _DeliveryCandidate:
    user: ChatMessage
    assistant: ChatMessage
    complete: bool
    remember: bool
    evicted: list[ChatMessage]


class ChatSession:
    """
    The conversation worker streams replies.
    Other threads may cancel them.
    A reply is added to history at once.
    An interrupted voice answer is removed again.
    """

    def __init__(
        self,
        model: StreamingChatModel,
        *,
        system_prompt: str,
        max_history_turns: int,
        context_tool: RequestContextTool | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._model = model
        self._max_history_turns = max_history_turns
        self._context_tool = context_tool
        self._memory = memory

        # cli.py replaces this after construction, because the reset command
        # needs this session's reset_conversation().
        self.commands = CommandRegistry()
        self._history: list[ChatMessage] = []
        self._cancel_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._cancel_epoch = 0
        self._pending_delivery: _DeliveryCandidate | None = None
        restored_memory = None

        if memory is not None:
            restored_memory = memory.context_for_chat()

        self._prompts = PromptBuilder(
            system_prompt,
            memory_context=restored_memory,
            max_prompt_tokens=model.prompt_budget,
            count_tokens=model.count_tokens,
        )

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(message.copy() for message in self._history)

    def reset_conversation(self) -> str:
        """
        saved memory stays
        """

        self._history.clear()
        self._pending_delivery = None

        if self._context_tool is not None:
            self._context_tool.reset()

        return "The current conversation is cleared. Saved memory is still available."

    def stream_reply(self, transcript: str) -> Iterator[str]:
        user_text = transcript.strip()
        if not user_text:
            raise ValueError("transcript must not be empty")

        with self._cancel_lock:
            epoch = self._cancel_epoch

        return self._stream_reply_iter(user_text, epoch)

    def cancel_current_reply(self) -> None:
        """
        cancels also one that has not started streaming yet
        """

        with self._cancel_lock:
            self._cancel_epoch += 1

            if self._active_cancel is not None:
                self._active_cancel.set()
                self._model.cancel_current_reply()

            if self._pending_delivery is not None:
                self._pending_delivery.complete = False

    def _stream_reply_iter(self, user_text: str, epoch: int) -> Iterator[str]:
        cancel = threading.Event()
        with self._cancel_lock:
            if epoch != self._cancel_epoch:
                return

            if self._active_cancel is not None:
                raise RuntimeError("a response is already being generated")

            self._active_cancel = cancel
            self._pending_delivery = None

        parts: list[str] = []
        complete = False
        command: CommandMatch | None = None
        model_reply: Generator[str, None, None] | None = None

        try:
            command = self.commands.match(user_text)

            if cancel.is_set():
                return

            if command is None:
                model_reply = self._start_model_reply(user_text, cancel)
                reply = model_reply
            else:
                reply = self._run_command(command, cancel)

            if reply is None:
                return

            for text in reply:
                if cancel.is_set():
                    return

                if text:
                    parts.append(text)
                    yield text

            answer = "".join(parts).strip()
            if not answer and not cancel.is_set():
                raise RuntimeError("the local model returned an empty response")

            complete = not cancel.is_set()
        except Exception:
            # After a cancellation, the stopped model or command may raise an exception.
            # That error is expected and not reported.
            if not cancel.is_set():
                raise
        finally:
            try:
                if model_reply is not None:
                    # Stop llama.cpp's stream when the loop above ended early.
                    model_reply.close()
            finally:
                self._record_provisional_turn(
                    cancel,
                    user_text,
                    "".join(parts).strip(),
                    complete=complete,
                    remember=command is None,
                )

    def _run_command(
        self, command: CommandMatch, cancel: threading.Event
    ) -> tuple[str] | None:
        """
        return None if the turn was cancelled.

        A network command runs on its own thread, so a cancellation frees the
        conversation worker at once. Other commands may change session state, so
        they run on the conversation worker.
        """

        if command.blocking:
            response = _run_until_cancelled(command, cancel)
        else:
            response = command.run()

        if response is None or cancel.is_set():
            return None

        return (response,)

    def _start_model_reply(
        self, user_text: str, cancel: threading.Event
    ) -> Generator[str, None, None] | None:
        """
        Used to build the prompt and start the model
        """

        context = None

        if self._context_tool is not None:
            context = self._context_tool.context_for_request(user_text)

        request = self._prompts.build(self._history, user_text, context)

        # cancel_current_reply() signals the model under this lock, so the signal
        # cannot arrive between this check and prepare_reply(), which clears it.
        with self._cancel_lock:
            if cancel.is_set():
                return None

            self._model.prepare_reply()

        return self._model.stream_reply(request)

    def _record_provisional_turn(
        self,
        cancel: threading.Event,
        user_text: str,
        response: str,
        *,
        complete: bool,
        remember: bool,
    ) -> None:
        """
        Calling this will add the reply to history.
        """

        with self._cancel_lock:
            if response:
                user = {"role": "user", "content": user_text}
                assistant = {"role": "assistant", "content": response}
                self._history.extend((user, assistant))
                message_limit = self._max_history_turns * 2
                excess = max(0, len(self._history) - message_limit)
                evicted = self._history[:excess]
                del self._history[:excess]
                self._pending_delivery = _DeliveryCandidate(
                    user=user,
                    assistant=assistant,
                    complete=complete and not cancel.is_set(),
                    remember=remember,
                    evicted=evicted,
                )

            if self._active_cancel is cancel:
                self._active_cancel = None

    def finalize_turn(self, delivered: bool) -> None:
        with self._cancel_lock:
            pending = self._pending_delivery
            self._pending_delivery = None
            if pending is None:
                return

            if not delivered:
                # Speech lags behind generation by whole sentences, and there are no
                # word timestamps to tell what was heard. Removing the whole turn is
                # better than keeping text the user may never have heard.
                kept_messages = [
                    message
                    for message in self._history
                    if message is not pending.user and message is not pending.assistant
                ]
                self._history = pending.evicted + kept_messages
                return

        if not pending.complete or not pending.remember or self._memory is None:
            return

        self._memory.record_complete_turn(
            pending.user["content"], pending.assistant["content"]
        )


def _run_until_cancelled(command: CommandMatch, cancel: threading.Event) -> str | None:
    """
    Run a command on its own thread. None is returned if the turn is cancelled.
    """

    finished = threading.Event()
    responses: list[str] = []
    errors: list[Exception] = []

    def run() -> None:
        try:
            responses.append(command.run())
        except Exception as exception:
            errors.append(exception)
        finally:
            finished.set()

    threading.Thread(target=run, name=f"Command {command.name}", daemon=True).start()
    while not finished.wait(_CANCEL_POLL_SECONDS):
        if cancel.is_set():
            return None

    if errors:
        raise errors[0]

    return responses[0]
