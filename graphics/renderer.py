"""
renderer.py

Provides drawing utilities for visualizing information on video frames.
"""

from __future__ import annotations

import cv2


class Renderer:
    """
    Responsible for drawing overlays on video frames.
    """

    @staticmethod
    def draw_fps(frame, fps: float) -> None:
        """
        Draw the current FPS on the frame.

        Args:
            frame: Video frame.
            fps: Current frames per second.
        """

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )