from dataclasses import dataclass, field
import time

from alerts.alert_level import AlertLevel
from alerts.alert_transition import AlertTransition


@dataclass
class AlertState:
    track_id: int
    level: AlertLevel
    transition: AlertTransition = AlertTransition.NONE
    entered_at: float = field(default_factory=time.time)
    confirmed: bool = False