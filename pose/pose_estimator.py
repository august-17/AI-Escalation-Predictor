"""
pose_estimator.py

Provides a YOLO Pose based pose estimator.
"""

from __future__ import annotations

import logging

from ultralytics import YOLO


logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Estimates human body keypoints using YOLO Pose.
    """

    def __init__(self, model_name: str = "yolov8n-pose.pt") -> None:
        """
        Initialize the pose estimator.

        Args:
            model_name: YOLO pose model filename.
        """

        logger.info("Loading pose model: %s", model_name)

        self.model = YOLO(model_name)

        logger.info("Pose model loaded successfully.")

    def estimate(self, frame):
        """
        Estimate poses in a frame.

        Args:
            frame: OpenCV frame.

        Returns:
            YOLO pose results.
        """

        return self.model(frame, verbose=False)