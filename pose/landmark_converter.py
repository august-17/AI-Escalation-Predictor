"""
Converts MediaPipe landmarks into image coordinates.
"""

from __future__ import annotations

from models.pose_landmark import PoseLandmark
from models.pose_result import PoseResult


class LandmarkConverter:
    """
    Utility for converting MediaPipe landmarks.
    """

    @staticmethod
    def convert(
        pose_results,
        roi_x: int,
        roi_y: int,
        roi_width: int,
        roi_height: int
    ) -> PoseResult:

        if (
            pose_results is None
            or pose_results.pose_landmarks is None
        ):
            return PoseResult()

        landmarks: list[PoseLandmark] = []

        for landmark in pose_results.pose_landmarks.landmark:

            normalized_x = min(max(landmark.x, 0.0), 1.0)
            normalized_y = min(max(landmark.y, 0.0), 1.0)

            x = int(normalized_x * roi_width) + roi_x
            y = int(normalized_y * roi_height) + roi_y

            landmarks.append(
                PoseLandmark(
                    x=x,
                    y=y,
                    visibility=landmark.visibility
                )
            )

        return PoseResult(landmarks=landmarks)