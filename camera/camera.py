"""
camera.py

Provides a reusable Camera class for accessing webcam or video sources.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2

from models.types import TrackPair
from config.settings import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)

logger = logging.getLogger(__name__)


class Camera:
    """
    Manages camera operations.

    Responsibilities:
        - Open camera
        - Read frames
        - Release camera

    This class deliberately does NOT:
        - Display frames
        - Calculate FPS
        - Perform AI inference
    """

    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        """
        Initialize a Camera object.

        Args:
            camera_index: Index of the webcam.
        """
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None


    def open(self) -> bool:
        """
        Open the camera.

        Returns:
            True if successful, False otherwise.
        """
        self.capture = cv2.VideoCapture(self.camera_index)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if self.capture is None or not self.capture.isOpened():
            logger.error("Unable to open camera %s.", self.camera_index)
            return False

        logger.info("Camera %s opened successfully.", self.camera_index)
        return True


    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """
        Read a frame from the camera.

        Returns:
            (success, frame)
        """
        if self.capture is None:
            logger.error("Camera has not been opened.")
            return False, None

        return self.capture.read()


    def release(self) -> None:
        """
        Release the camera.
        """
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            logger.info("Camera released.")


    def get_resolution(self) -> TrackPair:
        """
        Return the current camera resolution.

        Returns:
            A tuple containing (width, height).
        """
        if self.capture is None:
            return 0, 0

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return width, height