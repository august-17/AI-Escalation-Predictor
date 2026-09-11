"""YOLO + ByteTrack person tracking adapter."""

from __future__ import annotations

import logging
from typing import Any

import cv2

from config.settings import PERSON_CLASS_ID, TRACK_CONFIDENCE_THRESHOLD, TRACK_PERSIST
from models.tracked_person import TrackedPerson

logger = logging.getLogger(__name__)


class PersonTracker:
    def __init__(self, model_name: str = "yolov8n.pt", device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from error
        logger.info("Loading tracking model: %s", model_name)
        self.model: Any = YOLO(model_name)
        self.device = device

    def track(self, frame: cv2.typing.MatLike) -> list[TrackedPerson]:
        kwargs: dict[str, object] = {
            "persist": TRACK_PERSIST,
            "tracker": "bytetrack.yaml",
            "verbose": False,
            "classes": [PERSON_CLASS_ID],
            "conf": TRACK_CONFIDENCE_THRESHOLD,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        people: list[TrackedPerson] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue
                confidence = float(box.conf[0])
                if confidence < TRACK_CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                people.append(
                    TrackedPerson(int(box.id[0]), (x1, y1, x2, y2), confidence)
                )
        return people
