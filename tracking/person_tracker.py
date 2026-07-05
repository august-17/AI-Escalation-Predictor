"""
person_tracker.py

Provides a YOLO + ByteTrack based person tracker.
"""

from __future__ import annotations

import logging

from ultralytics import YOLO

from models.tracked_person import TrackedPerson


logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Tracks people across video frames using YOLO and ByteTrack.
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

    def track(self, frame) -> list[TrackedPerson]:
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
            persist=True,
            verbose=False,
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id != 0:
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                tracked_people.append(
                    TrackedPerson(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                    )
                )

        return tracked_people