"""
person_detector.py

Provides a YOLO-based person detector.
"""

from __future__ import annotations

import logging

import cv2

from ultralytics import YOLO

from models.person_detection import PersonDetection
from config.settings import PERSON_CLASS_ID
from models.types import BoundingBox

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Detects people in video frames using YOLO.
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        """
        Initialize the person detector.

        Args:
            model_name: YOLO model filename.
        """

        logger.info("Loading YOLO model: %s", model_name)

        self.model = YOLO(model_name)

        logger.info("YOLO model loaded successfully.")

    def detect(self, frame: cv2.typing.MatLike) -> list[PersonDetection]:
        """
        Detect people in a frame.

        Args:
            frame: OpenCV frame.

        Returns:
            A list of detected people in the current frame.
        """

        detections: list[PersonDetection] = []

        results = self.model(frame, verbose=False)

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                bbox: BoundingBox = (x1, y1, x2, y2)

                confidence = float(box.conf[0])

                detections.append(
                    PersonDetection(
                        bbox=bbox,
                        confidence=confidence,
                    )
                )

        return detections