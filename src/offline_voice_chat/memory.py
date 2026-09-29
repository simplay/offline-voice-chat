"""Bounded, local, cross-restart conversation memory."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_DEFAULT_MAX_ENTRIES = 24
_DEFAULT_MAX_ENTRY_CHARS = 280
_DEFAULT_MAX_CONTEXT_CHARS = 1_600
_MAX_STORAGE_BYTES = 256 * 1024
_ALLOWED_KINDS = frozenset({"fact", "preference", "decision", "commitment"})
# Model output is untrusted, so the assistant cannot create facts or preferences.
_ASSISTANT_KINDS = frozenset({"decision", "commitment"})
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_SMALL_TALK = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|okay|ok|yes|no|goodbye|bye)[.! ]*$",
    re.IGNORECASE,
)
_MEMORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "preference",
        re.compile(
            r"\b(?:i (?:prefer|like|love|dislike|hate|want|need)|"
            r"my preference(?: is|s are)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fact",
        re.compile(
            r"\b(?:my name is|i am|i'm|i live|i work|i have|"
            r"my [a-z][a-z -]{0,40} is|remember that)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "decision",
        re.compile(
            r"\b(?:i (?:decided|choose|chose|selected)|"
            r"we (?:decided|agreed|chose|selected)|"
            r"the decision is|we will use)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "commitment",
        re.compile(
            r"\b(?:i(?:'ll| will)|we(?:'ll| will)|you(?:'ll| will)|"
            r"next steps?|to-?do|follow[- ]?up|remember to)\b",
            re.IGNORECASE,
        ),
    ),
)


def default_memory_path(
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    environment = os.environ if environment is None else environment
    platform = sys.platform if platform_name is None else platform_name

    if platform.startswith("win"):
        local_app_data = environment.get("LOCALAPPDATA", "").strip()
        base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base / "offline-voice-chat" / "memory.json"

    if platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "offline-voice-chat"
            / "memory.json"
        )

    state_home = environment.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "offline-voice-chat" / "memory.json"


@dataclass(frozen=True)
class MemoryEntry:
    kind: str
    text: str


class PersistentConversationMemory:
    """
    The file is replaced atomically, so a crash never leaves a broken file.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_entry_chars: int = _DEFAULT_MAX_ENTRY_CHARS,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self._path = path.expanduser().resolve()
        self._max_entries = max_entries
        self._max_entry_chars = max_entry_chars
        self._max_context_chars = max_context_chars
        self._on_warning = on_warning
        self._warned: set[str] = set()
        self._entries: list[MemoryEntry] = []
        self._ongoing_user = ""
        self._ongoing_assistant = ""
        self._loaded = False
        self._write_allowed = True

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries)

    def load(self) -> bool:
        """
        Returns False when there is no file or it cannot be used.
        """

        self._entries = []
        self._ongoing_user = ""
        self._ongoing_assistant = ""
        self._loaded = True
        self._write_allowed = True

        try:
            with self._path.open("rb") as source:
                raw_bytes = source.read(_MAX_STORAGE_BYTES + 1)
        except FileNotFoundError:
            return False
        except OSError as exception:
            self._disable_writes(
                "read",
                f"could not read persistent memory at {self._path}: {exception}",
            )
            return False

        if len(raw_bytes) > _MAX_STORAGE_BYTES:
            self._disable_writes(
                "invalid",
                f"ignored invalid persistent memory at {self._path}: the "
                f"file exceeds {_MAX_STORAGE_BYTES} bytes; use --clear-memory "
                "to reset it",
            )
            return False

        # The file may be hand-edited or from another version. Any unexpected
        # shape fails here; the file is then ignored and never overwritten.
        try:
            text = raw_bytes.decode("utf-8")
            document = json.loads(text)
            entries, ongoing_user, ongoing_assistant = self._parse_document(document)
        except (AttributeError, KeyError, TypeError, ValueError) as exception:
            self._disable_writes(
                "invalid",
                f"ignored invalid persistent memory at {self._path}: {exception}; "
                "use --clear-memory to reset it",
            )
            return False

        self._entries = entries[-self._max_entries :]
        self._ongoing_user = self._truncate(ongoing_user)
        self._ongoing_assistant = self._truncate(ongoing_assistant)
        return True

    def context_for_chat(self) -> str | None:
        """
        Returns None when nothing is stored.
        """

        if not self._loaded:
            self.load()

        if not self._entries and not self._ongoing_user:
            return None

        header = "[BEGIN PERSISTENT MEMORY DATA]\n"
        footer = "\n[END PERSISTENT MEMORY DATA]"
        available = self._max_context_chars - len(header) - len(footer)
        selected: list[str] = []

        if self._ongoing_user:
            thread = f"Ongoing user topic: {self._ongoing_user}"

            if self._ongoing_assistant:
                thread += f"\nLast assistant outcome: {self._ongoing_assistant}"

            if len(thread) <= available:
                selected.append(thread)
                available -= len(thread) + 1

        entries_heading = "Remembered facts and decisions:\n"
        available -= len(entries_heading)
        retained_entries: list[str] = []

        for entry in reversed(self._entries):
            line = f"- {entry.kind}: {entry.text}"
            if len(line) + 1 > available:
                continue

            retained_entries.append(line)
            available -= len(line) + 1

        if retained_entries:
            selected.append(entries_heading + "\n".join(reversed(retained_entries)))

        if not selected:
            return None

        return header + "\n".join(selected) + footer

    def record_complete_turn(self, user_text: str, assistant_text: str) -> bool:
        """
        Returns whether the memory was saved.
        """

        if not self._loaded:
            self.load()

        user = _normalize_text(user_text)
        assistant = _normalize_text(assistant_text)
        if not user or not assistant:
            return False

        for entry in _extract_entries(user, assistant):
            compact = MemoryEntry(entry.kind, self._truncate(entry.text))
            key = (compact.kind, compact.text.casefold())
            self._entries = [
                current
                for current in self._entries
                if (current.kind, current.text.casefold()) != key
            ]
            self._entries.append(compact)

        self._entries = self._entries[-self._max_entries :]

        if _is_substantive(user):
            self._ongoing_user = self._truncate(user)
            self._ongoing_assistant = self._truncate(assistant)

        if not self._write_allowed:
            return False

        return self._write()

    def clear(self) -> bool:
        """
        A failure only warns, so startup continues.
        """

        self._entries = []
        self._ongoing_user = ""
        self._ongoing_assistant = ""
        self._loaded = True

        try:
            self._path.unlink(missing_ok=True)
        except OSError as exception:
            self._write_allowed = False
            self._warn_once(
                "clear",
                f"could not clear persistent memory at {self._path}: {exception}",
            )
            return False

        self._write_allowed = True
        return True

    def _parse_document(self, document: Any) -> tuple[list[MemoryEntry], str, str]:
        version = document["schema_version"]
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {version!r}")

        entries = [self._parse_entry(item) for item in document["entries"]]
        ongoing = document.get("ongoing_thread", {})
        ongoing_user = _normalize_text(ongoing.get("user", ""))
        ongoing_assistant = _normalize_text(ongoing.get("assistant", ""))
        return entries, ongoing_user, ongoing_assistant

    def _parse_entry(self, item: Any) -> MemoryEntry:
        kind = item["kind"]
        text = _normalize_text(item["text"])

        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"unknown memory kind {kind!r}")

        if not text:
            raise ValueError("a memory entry is empty")

        return MemoryEntry(kind, self._truncate(text))

    def _write(self) -> bool:
        document = {
            "schema_version": _SCHEMA_VERSION,
            "entries": [
                {"kind": entry.kind, "text": entry.text} for entry in self._entries
            ],
            "ongoing_thread": {
                "user": self._ongoing_user,
                "assistant": self._ongoing_assistant,
            },
        }
        temporary_path: Path | None = None

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                # mkstemp() creates this file readable only by its owner.
                temporary_path = Path(destination.name)
                json.dump(
                    document,
                    destination,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())

            os.replace(temporary_path, self._path)
            return True
        except OSError as exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            self._warn_once(
                "write",
                f"could not save persistent memory at {self._path}: {exception}; "
                "continuing with session memory only",
            )
            return False

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_entry_chars:
            return text

        return text[: self._max_entry_chars - 1].rstrip() + "…"

    def _disable_writes(self, category: str, message: str) -> None:
        self._write_allowed = False
        self._warn_once(category, message)

    def _warn_once(self, category: str, message: str) -> None:
        if category in self._warned:
            return

        self._warned.add(category)

        if self._on_warning is not None:
            self._on_warning(message)


def _extract_entries(user: str, assistant: str) -> list[MemoryEntry]:
    user_entries = _entries_from_speaker("User", user, _ALLOWED_KINDS)
    assistant_entries = _entries_from_speaker("Assistant", assistant, _ASSISTANT_KINDS)
    return user_entries + assistant_entries


def _entries_from_speaker(
    speaker: str, text: str, allowed_kinds: frozenset[str]
) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []

    for sentence in _SENTENCE_BOUNDARY.split(text):
        normalized = _normalize_text(sentence)
        if not normalized:
            continue

        for kind, pattern in _MEMORY_PATTERNS:
            if kind in allowed_kinds and pattern.search(normalized):
                entries.append(MemoryEntry(kind, f"{speaker}: {normalized}"))
                break

    return entries


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _is_substantive(text: str) -> bool:
    return len(text.split()) >= 3 and _SMALL_TALK.fullmatch(text) is None
