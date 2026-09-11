"""Command-line runtime options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from camera.camera import CaptureSource, parse_source
from config.settings import CAMERA_HEIGHT, CAMERA_WIDTH, YOLO_MODEL


@dataclass(slots=True)
class RuntimeConfig:
    source: CaptureSource = 0
    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT
    model: str = YOLO_MODEL
    device: str | None = None
    output: Path | None = None
    events_log: Path = Path("logs/events.jsonl")
    headless: bool = False
    disable_pose: bool = False
    max_frames: int | None = None

    @classmethod
    def from_values(
        cls,
        source: str | int = 0,
        output: str | None = None,
        events_log: str = "logs/events.jsonl",
        **kwargs: object,
    ) -> "RuntimeConfig":
        return cls(
            source=parse_source(source),
            output=Path(output) if output else None,
            events_log=Path(events_log),
            **kwargs,
        )
