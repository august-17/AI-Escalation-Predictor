"""
Tracks pose motion across video frames.
"""

from __future__ import annotations

import time

from models.pose_result import PoseResult


class MotionTracker:
    """
    Tracks pose history for each person.
    """

    def __init__(self) -> None:

        self._history: dict[int, tuple[PoseResult, float]] = {}


    def update(self, track_id: int, pose: PoseResult | None) -> None:
        """
        Store the latest pose for a tracked person.
        """

        if pose is None:
            return

        self._history[track_id] = (pose, time.perf_counter())


    def previous_pose(self, track_id: int) -> PoseResult | None:

        data = self._history.get(track_id)

        if data is None:
            return None

        return data[0]