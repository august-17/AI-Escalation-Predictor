"""
fps.py

Provides an FPSCounter class for measuring application performance.
"""

from __future__ import annotations

import time


class FPSCounter:
    """
    Measures the application's frames per second (FPS).
    """

    def __init__(self) -> None:
        """Initialize the FPS counter."""

        self._frame_count: int = 0
        self._fps: float = 0.0
        self._previous_time: float = time.perf_counter()


    def update(self) -> None:
        """
        Update the FPS calculation.

        Call this once for every processed frame.
        """

        self._frame_count += 1

        current_time = time.perf_counter()
        elapsed = current_time - self._previous_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._previous_time = current_time


    def get_fps(self) -> float:
        """
        Return the latest FPS value.
        """

        return self._fps