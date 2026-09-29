from __future__ import annotations

import unittest

import numpy as np
from fakes import (
    FakeClock,
    InstantOutputDevice,
)

from offline_voice_chat.audio import (
    RealtimePlayback,
    RealtimePlaybackError,
    StreamingResampler,
)

# 24 kHz mono PCM16 uses two bytes per sample.
ONE_SECOND_PCM16 = bytes(24_000 * 2)


class RealtimePlaybackTests(unittest.TestCase):
    def test_playback_is_bounded_by_duration_as_well_as_chunk_count(self) -> None:
        playback = RealtimePlayback(
            InstantOutputDevice(), np, device=None, max_buffer_seconds=0.1
        )

        with self.assertRaisesRegex(RealtimePlaybackError, "oversized"):
            playback.enqueue("item", 0, bytes(24000))

        playback.enqueue("item", 0, bytes(4000))
        with self.assertRaisesRegex(RealtimePlaybackError, "buffer limit"):
            playback.enqueue("item", 0, bytes(4000))

        playback.clear()
        playback.enqueue("new", 0, bytes(4000))

        playback = RealtimePlayback(
            InstantOutputDevice(), np, device=None, max_queue_chunks=1
        )
        playback.enqueue("item", 0, b"\x01\x00\x02\x00")

        with self.assertRaisesRegex(RealtimePlaybackError, "fell behind"):
            playback.enqueue("item", 0, b"\x03\x00\x04\x00")

    def test_truncation_discounts_audio_still_buffered_in_the_output_device(
        self,
    ) -> None:
        playback = RealtimePlayback(
            InstantOutputDevice(latency=0.1), np, device=None, clock=FakeClock(10.0)
        )
        playback.start()

        try:
            playback.enqueue("item", 0, ONE_SECOND_PCM16)
            # White-box: wait until the playback worker has written every chunk.
            playback._queue.join()
            self.assertEqual(playback.clear().audio_end_ms, 900)
        finally:
            playback.close()

    def test_a_completely_heard_item_needs_no_truncation(self) -> None:
        clock = FakeClock(10.0)
        playback = RealtimePlayback(
            InstantOutputDevice(latency=0.1), np, device=None, clock=clock
        )
        playback.start()

        try:
            playback.enqueue("answer", 0, ONE_SECOND_PCM16)
            playback.finish_item("answer", 0)
            playback._queue.join()
            clock.advance(0.2)  # The 100 ms output latency has passed.

            self.assertIsNone(playback.clear())
        finally:
            playback.close()

    def test_resampling_single_samples_preserves_phase_when_downsampling(self) -> None:
        for source_rate, target_rate in (
            (48000, 16000),
            (44100, 24000),
            (24000, 48000),
        ):
            with self.subTest(rates=(source_rate, target_rate)):
                source = np.linspace(-0.75, 0.75, 100, dtype=np.float32)
                whole_block_resampler = StreamingResampler(source_rate, target_rate, np)
                expected = whole_block_resampler.convert(source)
                resampler = StreamingResampler(source_rate, target_rate, np)
                single_sample_outputs = [
                    resampler.convert(source[index : index + 1]) for index in range(100)
                ]
                actual = np.concatenate(single_sample_outputs)
                np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
