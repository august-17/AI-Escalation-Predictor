"""
person_detector.py

Provides a YOLO-based person detector.
"""

from __future__ import annotations

import logging

from ultralytics import YOLO

from models.person_detection import PersonDetection


logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Detects people in video frames using YOLO.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        """
        Initialize the person detector.

        Args:
            model_name: YOLO model filename.
        """

        logger.info("Loading YOLO model: %s", model_name)

        self.model = YOLO(model_name)

        logger.info("YOLO model loaded successfully.")

    def detect(self, frame) -> list[dict]:
        """
        Detect people in a frame.

        Args:
            frame: OpenCV frame.

        Returns:
            List of detected people.
        """

        detections: list[PersonDetection] = []

        results = self.model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id != self.PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                detections.append(
                    PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                    )
                )

        return detections