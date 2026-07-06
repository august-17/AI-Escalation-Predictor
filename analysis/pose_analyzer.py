"""
Utilities for analyzing human pose.
"""

from __future__ import annotations

from models.pose_landmark import PoseLandmark
from models.pose_result import PoseResult


class PoseAnalyzer:
    """
    Extracts useful information from a pose.
    """

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14

    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    LEFT_HIP = 23
    RIGHT_HIP = 24

    @staticmethod
    def _landmark(pose: PoseResult, index: int) -> PoseLandmark | None:

        if index >= len(pose.landmarks):
            return None

        return pose.landmarks[index]
    

    @classmethod
    def left_wrist(cls, pose: PoseResult) -> PoseLandmark | None:

        return cls._landmark(pose, cls.LEFT_WRIST)
    

    @classmethod
    def right_wrist(cls, pose: PoseResult) -> PoseLandmark | None:

        return cls._landmark(pose, cls.RIGHT_WRIST)
    
    
    @classmethod
    def left_shoulder(cls, pose: PoseResult) -> PoseLandmark | None:

        return cls._landmark(pose, cls.LEFT_SHOULDER)
    

    @classmethod
    def right_shoulder(cls, pose: PoseResult) -> PoseLandmark | None:

        return cls._landmark(pose, cls.RIGHT_SHOULDER)
    

    @classmethod
    def body_center(cls, pose: PoseResult) -> PoseLandmark | None:
        """
        Compute the center of the upper body using the
        shoulder and hip landmarks.
        """

        left_shoulder = cls._landmark(pose, cls.LEFT_SHOULDER)

        right_shoulder = cls._landmark(pose, cls.RIGHT_SHOULDER)

        left_hip = cls._landmark(pose, cls.LEFT_HIP)

        right_hip = cls._landmark(pose, cls.RIGHT_HIP)

        landmarks = [
            landmark
            for landmark in (
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            )
            if landmark is not None
        ]

        if not landmarks:
            return None

        x = sum(landmark.x for landmark in landmarks) // len(landmarks)
        y = sum(landmark.y for landmark in landmarks) // len(landmarks)

        visibility = min(
            landmark.visibility
            for landmark in landmarks
        )

        return PoseLandmark(
            x=x,
            y=y,
            visibility=visibility
        )
    

    @classmethod
    def shoulder_center(cls, pose: PoseResult) -> PoseLandmark | None:
        """
        Compute the center point between the shoulders.
        """

        left = cls.left_shoulder(pose)
        right = cls.right_shoulder(pose)

        if left is None or right is None:
            return None

        return PoseLandmark(
            x=(left.x + right.x) // 2,
            y=(left.y + right.y) // 2,
            visibility=min(
                left.visibility,
                right.visibility
            )
        )
    
    
    @classmethod
    def hip_center(cls, pose: PoseResult) -> PoseLandmark | None:
        """
        Compute the center point between the hips.
        """

        left = cls._landmark(pose, cls.LEFT_HIP)

        right = cls._landmark(pose, cls.RIGHT_HIP)

        if left is None or right is None:
            return None

        return PoseLandmark(
            x=(left.x + right.x) // 2,
            y=(left.y + right.y) // 2,
            visibility=min(
                left.visibility,
                right.visibility
            )
        )