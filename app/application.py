"""
Application controller for the AI Escalation Predictor.
"""

from __future__ import annotations

import logging

import cv2

from camera.camera import Camera


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

    def run(self) -> None:
        """
        Start the application.
        """

        logger.info("Starting AI Escalation Predictor...")

        if not self.camera.open():
            logger.error("Failed to initialize camera.")
            return

        logger.info("Press 'Q' to quit.")

        try:

            while True:

                success, frame = self.camera.read()

                if not success:
                    logger.error("Unable to read frame.")
                    break

                cv2.imshow("AI Escalation Predictor", frame)

                key = cv2.waitKey(1)

                if key == ord("q") or key == ord("Q"):
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