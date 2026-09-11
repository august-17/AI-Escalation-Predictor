from __future__ import annotations

from dataclasses import dataclass

from alerts.alert_level import AlertLevel
from alerts.alert_transition import AlertTransition


@dataclass(slots=True)
class AlertState:
    track_id: int
    level: AlertLevel = AlertLevel.NORMAL
    transition: AlertTransition = AlertTransition.NONE
    entered_at: float = 0.0
    confirmed: bool = False
    event_created: bool = False
    current_risk: float = 0.0
