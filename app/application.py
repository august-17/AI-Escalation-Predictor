"""
Application controller for the AI Escalation Predictor.
"""

from __future__ import annotations

import logging

import cv2

from camera.camera import Camera
from camera.fps import FPSCounter
from graphics.renderer import Renderer
from detection.person_detector import PersonDetector


logger = logging.getLogger(__name__)


class Application:
    """
    Coordinates all application components.
    """

    def __init__(self) -> None:
        """
        Initialize the application.
        """
        self.camera = Camera()
        self.fps_counter = FPSCounter()
        self.detector = PersonDetector()

    def run(self) -> None:
        """
        Start the application.
        """

        logger.info("Starting AI Escalation Predictor...")

        if not self.camera.open():
            logger.error("Failed to initialize camera.")
            return
        
        width, height = self.camera.get_resolution()
        logger.info("Camera resolution: %d x %d", width, height)

        logger.info("Press 'Q' to quit.")

        try:
            while True:

                success, frame = self.camera.read()

                if not success:
                    logger.error("Unable to read frame.")
                    break

                detections = self.detector.detect(frame)

                Renderer.draw_detections(frame, detections)

                self.fps_counter.update()

                fps = self.fps_counter.get_fps()
                
                Renderer.draw_fps(frame, fps)

                cv2.imshow("AI Escalation Predictor", frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q")):
                    logger.info("Exit requested by user.")
                    break

        except KeyboardInterrupt:
            logger.info("Application interrupted by user.")

        except Exception:
            logger.exception("An unexpected error occurred.")

        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            logger.info("Application closed.")