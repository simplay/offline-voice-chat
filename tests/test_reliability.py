from __future__ import annotations

import unittest

from fakes import FakeClock

from offline_voice_chat.reliability import ActiveTurnDeadline


class ActiveTurnDeadlineTests(unittest.TestCase):
    def test_activity_refreshes_only_the_matching_phase(self) -> None:
        clock = FakeClock()
        deadline = ActiveTurnDeadline(clock=clock)
        deadline.arm("generation", 5.0)
        clock.advance(4.0)

        self.assertFalse(deadline.touch("playback"))
        self.assertTrue(deadline.touch("generation"))
        self.assertIsNone(deadline.poll(now=8.9))
        timeout_signal = deadline.poll(now=9.0)

        self.assertIsNotNone(timeout_signal)
        self.assertEqual(timeout_signal.kind, "timeout")
        self.assertEqual(timeout_signal.phase, "generation")
        self.assertIsNone(deadline.phase)


if __name__ == "__main__":
    unittest.main()
