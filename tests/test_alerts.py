from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alerts.alert_level import AlertLevel
from alerts.alert_manager import AlertManager
from alerts.event_logger import EventLogger
from models.risk_breakdown import RiskBreakdown


class AlertManagerTests(unittest.TestCase):
    def test_alert_requires_sustained_score_and_contains_risk(self) -> None:
        manager = AlertManager()
        state = manager.update({7: 0.82}, timestamp=0.0)[7]
        self.assertEqual(state.level, AlertLevel.CRITICAL)
        self.assertFalse(state.confirmed)
        details = {7: RiskBreakdown(proximity=0.18, raw_total=0.82, smoothed_total=0.82)}
        state = manager.update({7: 0.82}, details, timestamp=0.8)[7]
        self.assertTrue(state.confirmed)
        events = manager.get_new_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].risk, 0.82)
        self.assertEqual(events[0].dominant_factor, "proximity")
        self.assertEqual(manager.get_new_events(), [])

    def test_hysteresis_prevents_threshold_flapping(self) -> None:
        manager = AlertManager()
        self.assertEqual(manager.update({1: 0.51}, timestamp=0.0)[1].level, AlertLevel.WARNING)
        self.assertEqual(manager.update({1: 0.48}, timestamp=0.1)[1].level, AlertLevel.WARNING)
        self.assertEqual(manager.update({1: 0.40}, timestamp=0.2)[1].level, AlertLevel.WATCH)

    def test_direct_level_jumps_have_meaningful_transition(self) -> None:
        manager = AlertManager()
        state = manager.update({1: 0.9}, timestamp=0.0)[1]
        self.assertEqual(state.transition.value, "Enter Critical")
        state = manager.update({1: 0.1}, timestamp=1.0)[1]
        self.assertEqual(state.transition.value, "Return to Normal")

    def test_event_logger_writes_valid_jsonl(self) -> None:
        manager = AlertManager()
        manager.update({2: 0.8}, timestamp=0.0)
        manager.update({2: 0.8}, timestamp=1.0)
        event = manager.get_new_events()[0]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "events.jsonl"
            EventLogger(target).write(event)
            payload = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(payload["track_id"], 2)
        self.assertEqual(payload["level"], "Critical")
        self.assertIn("timestamp_iso", payload)


if __name__ == "__main__":
    unittest.main()
