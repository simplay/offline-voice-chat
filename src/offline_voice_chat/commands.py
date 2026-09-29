"""
Commands match only the user's transcript, never model output, file text, or memory.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial

ONLINE_COMMAND_PREFIX = "please check online"

# voice.py handles these phrases itself so they can interrupt active work. They
# live here so the help text and the voice layer accept the same words.
STOP_PHRASES = ("stop", "stop speaking", "please stop", "cancel")
TERMINATE_PHRASES = ("please terminate yourself", "terminate yourself")


def normalize_command(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def parse_prefix_request(transcript: str, prefix: str) -> str | None:
    """
    Return the text after a spoken prefix
    """

    prefix_words = normalize_command(prefix).split()
    words = list(re.finditer(r"\w+", transcript))
    leading_words = words[: len(prefix_words)]
    spoken_prefix = [word.group().casefold() for word in leading_words]

    if spoken_prefix != prefix_words:
        return None

    prefix_end = leading_words[-1].end()
    return transcript[prefix_end:].strip(" .,:;!?-")


@dataclass(frozen=True)
class Command:
    name: str
    phrases: tuple[str, ...]
    description: str
    execute: Callable[[], str]


@dataclass(frozen=True)
class PrefixCommand:
    name: str
    prefix: str
    description: str
    execute: Callable[[str], str]
    # A blocking handler waits on the network. ChatSession runs it on its own
    # thread so a cancelled turn does not wait for it.
    # It must not change session state.
    blocking: bool = False


@dataclass(frozen=True)
class CommandMatch:
    """
    The command selected for one transcript
    """

    name: str
    handler: Callable[[], str]
    blocking: bool = False

    def run(self) -> str:
        response = self.handler()
        if not response.strip():
            raise RuntimeError(f"command {self.name!r} returned no response")

        return response


class CommandRegistry:
    """
    Register commands at startup. Handlers run on the conversation worker, so
    they must be fast. Register slow network commands as blocking.
    """

    def __init__(self, commands: Iterable[Command] = ()) -> None:
        self._commands: dict[str, Command] = {}
        self._phrases: dict[str, Command] = {}
        self._prefixes: dict[str, PrefixCommand] = {}

        for command in commands:
            self.register(command)

    def register(self, command: Command) -> None:
        self._check_new_name(command.name)
        phrases = tuple(normalize_command(phrase) for phrase in command.phrases)
        has_empty_phrase = not phrases or "" in phrases
        if has_empty_phrase:
            raise ValueError("commands need at least one non-empty phrase")

        repeats_a_phrase = len(set(phrases)) != len(phrases)
        phrase_taken = any(
            phrase in self._phrases or phrase in self._prefixes for phrase in phrases
        )
        if repeats_a_phrase or phrase_taken:
            raise ValueError(f"ambiguous command phrases: {command.name}")

        self._commands[command.name] = command
        self._phrases.update((phrase, command) for phrase in phrases)

    def register_prefix(self, command: PrefixCommand) -> None:
        self._check_new_name(command.name)
        prefix = normalize_command(command.prefix)
        overlaps_a_prefix = any(
            prefix == existing
            or prefix.startswith(existing + " ")
            or existing.startswith(prefix + " ")
            for existing in self._prefixes
        )
        if not prefix or prefix in self._phrases or overlaps_a_prefix:
            raise ValueError(f"ambiguous command prefix: {command.prefix!r}")

        self._prefixes[prefix] = command

    def match(self, transcript: str) -> CommandMatch | None:
        """
        Finds the command without running it.
        """

        command = self._phrases.get(normalize_command(transcript))
        if command is not None:
            return CommandMatch(command.name, command.execute)

        for prefix, prefix_command in self._prefixes.items():
            request = parse_prefix_request(transcript, prefix)
            if request is None:
                continue

            return CommandMatch(
                prefix_command.name,
                partial(prefix_command.execute, request),
                blocking=prefix_command.blocking,
            )

        return None

    def help_text(self) -> str:
        descriptions = [
            f"{command.phrases[0]} to {command.description}"
            for command in self._commands.values()
        ]
        descriptions.extend(
            f"{command.prefix} ... to {command.description}"
            for command in self._prefixes.values()
        )
        return (
            "You can say: "
            + "; ".join(descriptions)
            + f". Say {STOP_PHRASES[0]} to interrupt speech, "
            + f"or {TERMINATE_PHRASES[0]} to exit."
        )

    def _check_new_name(self, name: str) -> None:
        prefix_names = {command.name for command in self._prefixes.values()}
        if name in self._commands or name in prefix_names:
            raise ValueError(f"duplicate command name: {name!r}")


def default_commands(
    *,
    reset_conversation: Callable[[], str],
    now: Callable[[], datetime] = datetime.now,
    check_online: Callable[[str], str] | None = None,
    read_goodnight_story: Callable[[], str] | None = None,
) -> CommandRegistry:
    registry = CommandRegistry()
    for command in (
        Command(
            "time",
            ("what time is it", "tell me the time"),
            "hear the local time",
            lambda: f"It is {now():%H:%M}.",
        ),
        Command(
            "date",
            ("what is today's date", "what is the date"),
            "hear today's date",
            lambda: f"Today is {now():%A, %d %B %Y}.",
        ),
        Command(
            "datetime",
            ("what is the current date and time", "what is the current datetime"),
            "hear the local date and time",
            lambda: f"It is {now():%A, %d %B %Y at %H:%M}.",
        ),
        Command(
            "reset",
            ("clear conversation", "start a new conversation"),
            "clear the current conversation",
            reset_conversation,
        ),
        Command(
            "help",
            ("what can you do", "list commands", "help"),
            "list commands",
            registry.help_text,
        ),
    ):
        registry.register(command)

    if check_online is not None:
        registry.register_prefix(
            PrefixCommand(
                "online",
                ONLINE_COMMAND_PREFIX,
                "search for the following weather, news, or other query",
                check_online,
                blocking=True,
            )
        )

    if read_goodnight_story is not None:
        registry.register(
            Command(
                "goodnight_story",
                (
                    "read the goodnight story",
                    "please read the goodnight story",
                    "tell me the goodnight story",
                ),
                "hear the short goodnight story",
                read_goodnight_story,
            )
        )

    return registry
