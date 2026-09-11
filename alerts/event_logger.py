"""Append-only JSON Lines logging for confirmed research alerts."""

from __future__ import annotations

import json
from pathlib import Path

from alerts.alert_event import AlertEvent


class EventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, event: AlertEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            json.dump(event.as_dict(), stream, sort_keys=True)
            stream.write("\n")
