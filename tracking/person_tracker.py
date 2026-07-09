"""
person_tracker.py

Provides a YOLO + ByteTrack based person tracker.
"""

from __future__ import annotations

import logging

import cv2

from ultralytics import YOLO

from models.tracked_person import TrackedPerson
from models.types import BoundingBox
from config.settings import (
    PERSON_CLASS_ID,
    TRACK_PERSIST,
    TRACK_CONFIDENCE_THRESHOLD
)


logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Detects and tracks people across video frames using YOLOv8 and ByteTrack.
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        """
        Initialize the tracker.

        Args:
            model_name: YOLO model filename.
        """

        logger.info("Loading tracking model: %s", model_name)

        self.model = YOLO(model_name)

        logger.info("Tracking model loaded successfully.")


    def track(self, frame: cv2.typing.MatLike) -> list[TrackedPerson]:
        """
        Track people in a frame.

        Args:
            frame: OpenCV frame.

        Returns:
            List of tracked people.
        """

        tracked_people: list[TrackedPerson] = []

        results = self.model.track(
            frame,
            persist=TRACK_PERSIST,
            tracker="bytetrack.yaml",
            verbose=False
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id != PERSON_CLASS_ID:
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                bbox: BoundingBox = (x1, y1, x2, y2)

                confidence = float(box.conf[0])

                if confidence < TRACK_CONFIDENCE_THRESHOLD:
                    continue

                print(
                    f"ID {track_id} | Confidence: {confidence:.2f}"
                )

                tracked_people.append(
                    TrackedPerson(
                        track_id=track_id,
                        bbox=bbox,
                        confidence=confidence
                    )
                )

        return tracked_people