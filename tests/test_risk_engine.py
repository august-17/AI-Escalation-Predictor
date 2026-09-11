from __future__ import annotations

import unittest

from analysis.risk_engine import RiskEngine
from tests.helpers import person


class RiskEngineTests(unittest.TestCase):
    def test_single_stationary_person_remains_low(self) -> None:
        engine = RiskEngine()
        scores, details = engine.compute([person(1, 300)], timestamp=0.0)
        self.assertEqual(scores[1], 0.0)
        scores, details = engine.compute([person(1, 300)], timestamp=0.1)
        self.assertLess(scores[1], 0.05)
        self.assertEqual(details[1].dominant_factor, "none")

    def test_distance_is_normalized_by_person_height(self) -> None:
        engine = RiskEngine()
        small = [person(1, 200, height=100, with_pose=False), person(2, 350, height=100, with_pose=False)]
        _, small_details = engine.compute(small, timestamp=0.0)
        engine = RiskEngine()
        large = [person(1, 200, height=200, with_pose=False), person(2, 500, height=200, with_pose=False)]
        _, large_details = engine.compute(large, timestamp=0.0)
        self.assertAlmostEqual(small_details[1].proximity, large_details[1].proximity, places=6)

    def test_approach_and_fast_hands_raise_explainable_cues(self) -> None:
        engine = RiskEngine()
        engine.compute([person(1, 150), person(2, 550)], timestamp=0.0)
        scores, details = engine.compute(
            [person(1, 260, wrist_offset=80, extended=True), person(2, 440, wrist_offset=-80, extended=True)],
            timestamp=0.2,
        )
        self.assertGreater(details[1].approach_speed, 0.0)
        self.assertGreater(details[1].hand_speed, 0.0)
        self.assertGreater(details[1].arm_extension, 0.0)
        self.assertGreater(scores[1], 0.25)
        self.assertLessEqual(scores[1], 1.0)

    def test_close_interaction_accumulates_then_resets(self) -> None:
        engine = RiskEngine()
        pair = [person(1, 220), person(2, 400)]
        engine.compute(pair, timestamp=0.0)
        _, details = engine.compute(pair, timestamp=5.0)
        self.assertGreater(details[1].interaction_duration, 0.0)
        far = [person(1, 100), person(2, 700)]
        _, details = engine.compute(far, timestamp=5.2)
        self.assertEqual(details[1].interaction_duration, 0.0)

    def test_scores_decay_instead_of_dropping_abruptly(self) -> None:
        engine = RiskEngine()
        engine.compute([person(1, 150), person(2, 550)], timestamp=0.0)
        high, _ = engine.compute([person(1, 260, wrist_offset=80), person(2, 440)], timestamp=0.2)
        low, details = engine.compute([person(1, 260), person(2, 700)], timestamp=0.3)
        self.assertGreater(high[1], 0.0)
        self.assertGreater(low[1], details[1].raw_total)

    def test_missing_pose_still_supports_proximity(self) -> None:
        engine = RiskEngine()
        _, details = engine.compute(
            [person(1, 200, with_pose=False), person(2, 310, with_pose=False)],
            timestamp=0.0,
        )
        self.assertGreater(details[1].proximity, 0.0)
        self.assertEqual(details[1].pose_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
