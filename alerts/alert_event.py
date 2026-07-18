from dataclasses import dataclass, field
import time

from alerts.alert_level import AlertLevel


@dataclass
class AlertEvent:
    track_id: int
    level: AlertLevel
    risk: float
    timestamp: float = field(default_factory=time.time)