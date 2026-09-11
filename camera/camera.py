"""OpenCV capture abstraction supporting webcams and video files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeAlias

import cv2

from config.settings import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH

logger = logging.getLogger(__name__)
CaptureSource: TypeAlias = int | str


def parse_source(value: str | int | None) -> CaptureSource:
    """Convert a CLI source into a webcam index or a video path."""
    if value is None:
        return CAMERA_INDEX
    if isinstance(value, int):
        return value
    stripped = value.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped


class Camera:
    """Manage a webcam or video-file capture without doing inference."""

    def __init__(
        self,
        source: CaptureSource = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str)

    def open(self) -> bool:
        if self.is_file and not Path(self.source).expanduser().is_file():
            logger.error("Video source does not exist: %s", self.source)
            return False
        source = str(Path(self.source).expanduser()) if self.is_file else self.source
        self.capture = cv2.VideoCapture(source)
        if not self.is_file:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.capture.isOpened():
            logger.error("Unable to open capture source %s.", self.source)
            self.release()
            return False
        logger.info("Capture source %s opened successfully.", self.source)
        return True

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        if self.capture is None:
            logger.error("Capture source has not been opened.")
            return False, None
        return self.capture.read()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def get_resolution(self) -> tuple[int, int]:
        if self.capture is None:
            return 0, 0
        return (
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_fps(self) -> float:
        if self.capture is None:
            return 0.0
        value = float(self.capture.get(cv2.CAP_PROP_FPS))
        return value if value > 0 else 0.0
