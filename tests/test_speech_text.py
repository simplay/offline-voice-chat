import unittest

from fakes import FakeClock

from offline_voice_chat.speech_text import OpeningSpeechBuffer, is_probable_speaker_echo


def _released_text(streamed: str, seconds_per_piece: float) -> str | None:
    """Return the text released early from "|"-separated pieces at a steady pace."""

    pieces = streamed.split("|")
    clock = FakeClock()
    opening = OpeningSpeechBuffer(clock=clock)

    for index, piece in enumerate(pieces):
        clock.advance(seconds_per_piece)

        # voice.py asks before it pushes each piece.
        if opening.should_flush(piece):
            return "".join(pieces[:index])

    return None


class SpeechTextTests(unittest.TestCase):
    def test_opening_release_never_splits_a_number(self):
        # Qwen streams digits, separators, and the space before digits as
        # separate pieces.
        for streamed, released in (
            (
                "The population of Tokyo is about| |1|4|,|0|0|0|,|0|0|0| people|.",
                "The population of Tokyo is about 14,000,000",
            ),
            (
                "Die Temperatur beträgt heute| |2|1|,|5| Grad|.",
                "Die Temperatur beträgt heute 21,5",
            ),
            (
                "La population de la ville est de| |1|4| |0|0|0| habitants|.",
                "La population de la ville est de 14 000",
            ),
            ("The last train of the day leaves at| |1|0|:|3|0|.", None),
        ):
            with self.subTest(streamed=streamed):
                self.assertEqual(_released_text(streamed, 0.1), released)

    def test_probable_speaker_echo_is_ignored(self) -> None:
        self.assertTrue(
            is_probable_speaker_echo(
                "The answer being spoken through the speakers",
                "The answer being spoken through the speakers is still going.",
            )
        )
        self.assertFalse(
            is_probable_speaker_echo(
                "Please answer a different question",
                "The answer being spoken through the speakers is still going.",
            )
        )
        self.assertFalse(is_probable_speaker_echo("yeah", "Yeah, that is correct."))
