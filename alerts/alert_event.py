"""Serializable alert event model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time

from alerts.alert_level import AlertLevel


@dataclass(slots=True)
class AlertEvent:
    track_id: int
    level: AlertLevel
    risk: float
    dominant_factor: str = "unknown"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["level"] = self.level.value
        payload["timestamp_iso"] = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).isoformat()
        return payload
