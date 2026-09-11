"""MediaPipe pose estimator with one temporal estimator per track."""

from __future__ import annotations

from typing import Any

import cv2


class PoseEstimator:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from error

        solutions = getattr(mp, "solutions", None)
        pose_api = getattr(solutions, "pose", None)
        if pose_api is None:
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "This project uses the MediaPipe Legacy Pose API, but the installed "
                f"MediaPipe {version} does not provide mp.solutions. Install the "
                "supported release with: python -m pip install --force-reinstall "
                "--no-cache-dir mediapipe==0.10.21"
            )
        self._pose_api = pose_api
        self._instances: dict[int, Any] = {}
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def _create_pose(self) -> Any:
        return self._pose_api.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

    def estimate(self, track_id: int, person_roi: cv2.typing.MatLike) -> Any:
        if person_roi.size == 0:
            return None
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose = self._instances.setdefault(track_id, self._create_pose())
        return pose.process(rgb_roi)

    def remove_inactive(self, active_ids: set[int]) -> None:
        for track_id in set(self._instances) - active_ids:
            self._instances.pop(track_id).close()

    def close(self) -> None:
        for pose in self._instances.values():
            pose.close()
        self._instances.clear()
