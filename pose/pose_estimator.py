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
    Estimates human pose landmarks using MediaPipe.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:

        self._mp_pose = mp.solutions.pose

        self.pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info("MediaPipe Pose initialized successfully.")


    def estimate(self, person_roi: cv2.typing.MatLike):
        """
        Estimate pose landmarks for a cropped person image.

        Args:
            person_roi: Cropped image containing a single person.

        Returns:
            MediaPipe pose estimation results,
            or None if the ROI is empty.
        """

        if person_roi.size == 0:
            return None

        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

        return self.pose.process(rgb_roi)