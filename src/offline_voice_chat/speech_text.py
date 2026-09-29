"""Speech text rules: speaker-echo detection and early start of speech."""

import re
import time
from collections.abc import Callable
from difflib import SequenceMatcher

_SENTENCE_BOUNDARY = re.compile(r"[.!?][\"')\]]*(?:\s|$)")
_CLAUSE_END = re.compile(r"[,;:][\"')\]]*\s*$")
_MIN_RELEASE_CHARS = 28
_MAX_WAIT_CHARS = 120
_MAX_WAIT_SECONDS = 0.35
_BUFFER_CHARS = 256


class OpeningSpeechBuffer:
    """
    Starts speech with the first clause instead of waiting for the whole first
    sentence. It never splits a word or a number like 14,000.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started: float | None = None
        self._text = ""
        self._released = False

    def should_flush(self, next_text: str) -> bool:
        if self._released:
            return False

        now = self._clock()

        if self._started is None:
            self._started = now

        buffered_text = self._text
        self._text = (buffered_text + next_text)[-_BUFFER_CHARS:]

        if _SENTENCE_BOUNDARY.search(buffered_text):
            self._released = True
            return False  # The native synthesizer releases full sentences itself.

        # A piece of only whitespace cannot show whether a number continues.
        if not next_text.strip():
            return False

        between_words = buffered_text[-1:].isspace() or next_text[:1].isspace()
        last_char = buffered_text.rstrip()[-1:]
        next_char = next_text.lstrip()[:1]
        inside_number = last_char.isdigit() and next_char.isdigit()
        long_enough = len(buffered_text) >= _MIN_RELEASE_CHARS
        at_clause_end = _CLAUSE_END.search(buffered_text) is not None
        too_long = len(buffered_text) >= _MAX_WAIT_CHARS
        waited_too_long = now - self._started >= _MAX_WAIT_SECONDS
        ready = at_clause_end or too_long or waited_too_long

        if between_words and not inside_number and long_enough and ready:
            self._released = True
            return True

        return False


def is_probable_speaker_echo(transcript: str, spoken_text: str) -> bool:
    # Bound SequenceMatcher's worst-case work on the microphone event thread.
    normalized_transcript = _normalized_words(transcript[-512:])
    normalized_spoken = _normalized_words(spoken_text[-2048:])
    if len(normalized_transcript) < 5 or not normalized_spoken:
        return False

    # A first partial transcript is commonly only one word. Treat it as echo
    # only when it matches the beginning of what is currently being spoken;
    # arbitrary short words such as "stop" must still interrupt immediately.
    if len(normalized_transcript) < 12:
        return (
            normalized_spoken == normalized_transcript
            or normalized_spoken.startswith(f"{normalized_transcript} ")
        )

    if normalized_transcript in normalized_spoken:
        return True

    transcript_words = normalized_transcript.split()
    spoken_words = normalized_spoken.split()
    matcher = SequenceMatcher(None, transcript_words, spoken_words, autojunk=False)
    match = matcher.find_longest_match()
    matched_share = match.size / len(transcript_words)
    return matched_share >= 0.75


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))
