"""Local GGUF inference, streaming, and cooperative native cancellation."""

from __future__ import annotations

import threading
from collections.abc import Generator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from .contracts import ChatMessage

# Prompt room that the context window must keep after the answer's max_tokens.
MIN_PROMPT_TOKENS = 256

# Kept out of the prompt budget because PromptBuilder only estimates the
# chat template's framing tokens.
_CONTEXT_RESERVE_TOKENS = 64


def _configure_answer_only_chat(model: Any) -> None:
    """
    Keeps thinking models from speaking their reasoning.
    """

    # llama.cpp keeps the GGUF's own template here; guessed or built-in chat
    # formats have no enable_thinking switch. There is no public accessor.
    template_handler = model._chat_handlers.get(model.chat_format)

    if template_handler is None:
        return

    def answer_only_handler(**kwargs: Any) -> Any:
        kwargs["enable_thinking"] = False
        return template_handler(**kwargs)

    # create_chat_completion() prefers chat_handler over the template handler.
    model.chat_handler = answer_only_handler


class LlamaCppChatModel:
    def __init__(
        self,
        model_path: Path,
        *,
        context_size: int,
        max_tokens: int,
        temperature: float,
        threads: int | None,
        gpu_layers: int,
    ) -> None:
        try:
            from llama_cpp import (
                Llama,
                ggml_abort_callback,
                llama_set_abort_callback,
                llama_supports_gpu_offload,
            )
        except ImportError as exception:
            raise RuntimeError(
                "llama-cpp-python is not installed; install this project first"
            ) from exception

        self.prompt_budget = context_size - max_tokens - _CONTEXT_RESERVE_TOKENS
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._abort_event = threading.Event()
        self._cached_token_count = lru_cache(maxsize=128)(self._count_tokens)
        self.gpu_offload_available = llama_supports_gpu_offload()

        model_options: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": context_size,
            "n_gpu_layers": gpu_layers,
            # A CUDA/Metal build must also honor an explicit CPU-only request
            # for attention and auxiliary operations, not just model weights.
            "offload_kqv": gpu_layers != 0,
            "op_offload": gpu_layers != 0,
            "verbose": False,
        }

        if threads is not None:
            model_options["n_threads"] = threads

        self._model = Llama(**model_options)
        _configure_answer_only_chat(self._model)

        # llama.cpp checks this between decode steps. ctypes callbacks must
        # stay referenced while native code may call them.
        self._abort_callback = ggml_abort_callback(lambda _: self._abort_event.is_set())
        llama_set_abort_callback(self._model.ctx, self._abort_callback, None)

    def count_tokens(self, text: str) -> int:
        """
        Cached, because the history is counted again every turn.
        """

        return self._cached_token_count(text)

    def _count_tokens(self, text: str) -> int:
        token_ids = self._model.tokenize(text.encode("utf-8"), add_bos=False)
        return len(token_ids)

    def close(self) -> None:
        """
        Call only after the conversation worker has stopped.
        """

        self._cached_token_count.cache_clear()
        self._model.close()

    def prepare_reply(self) -> None:
        self._abort_event.clear()

    def cancel_current_reply(self) -> None:
        """
        llama.cpp stops at its next decode step.
        """

        self._abort_event.set()

    def stream_reply(
        self, messages: Sequence[ChatMessage]
    ) -> Generator[str, None, None]:
        if self._abort_event.is_set():
            return

        chunks = self._model.create_chat_completion(
            messages=list(messages),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )

        try:
            for chunk in chunks:
                if self._abort_event.is_set():
                    return

                # Each streamed chunk has one choice. Its delta holds the role
                # first, then content, and is empty in the final chunk.
                choice = chunk["choices"][0]
                text = choice["delta"].get("content")

                if text:
                    yield text

                if choice["finish_reason"] == "length":
                    raise RuntimeError(
                        "the answer reached --max-tokens before completion; "
                        "request a shorter answer or increase --max-tokens"
                    )
        finally:
            chunks.close()
