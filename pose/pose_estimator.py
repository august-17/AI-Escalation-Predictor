"""
pose_estimator.py

Provides a MediaPipe-based human pose estimator.
"""

from __future__ import annotations

import logging

import cv2
import mediapipe as mp


logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Estimates human body landmarks using MediaPipe Pose.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initialize the pose estimator.

        Args:
            min_detection_confidence:
                Minimum confidence required for pose detection.

            min_tracking_confidence:
                Minimum confidence required for landmark tracking.
        """

        self._mp_pose = mp.solutions.pose

        self.pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        logger.info("MediaPipe Pose initialized successfully.")

    def estimate(self, frame):
        """
        Estimate pose landmarks for a frame.

        Args:
            frame: OpenCV frame.

        Returns:
            MediaPipe pose results.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return self.pose.process(rgb_frame)