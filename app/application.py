"""
Application controller for the AI Escalation Predictor.
"""

from __future__ import annotations

import logging

import cv2

from camera.camera import Camera
from camera.fps import FPSCounter
from graphics.renderer import Renderer
from tracking.person_tracker import PersonTracker
from pose.pose_estimator import PoseEstimator
from config.settings import (
    POSE_INTERVAL_SINGLE_PERSON,
    POSE_INTERVAL_FEW_PEOPLE,
    POSE_INTERVAL_MANY_PEOPLE,
    MIN_PERSON_WIDTH,
    MIN_PERSON_HEIGHT
)


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
        self.tracker = PersonTracker()
        self.pose_estimator = PoseEstimator()

        self.frame_count = 0
        self.pose_cache = {}

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

                tracked_people = self.tracker.track(frame)

                self.frame_count += 1

                people_count = len(tracked_people)

                if people_count == 0:

                    run_pose = False

                elif people_count == 1:

                    run_pose = (
                        self.frame_count
                        % POSE_INTERVAL_SINGLE_PERSON
                        == 0
                    )

                elif people_count <= 3:

                    run_pose = (
                        self.frame_count
                        % POSE_INTERVAL_FEW_PEOPLE
                        == 0
                    )

                else:

                    run_pose = (
                        self.frame_count
                        % POSE_INTERVAL_MANY_PEOPLE
                        == 0
                    )

                for person in tracked_people:

                    x1, y1, x2, y2 = person.bbox

                    width = x2 - x1
                    height = y2 - y1

                    if (
                        width < MIN_PERSON_WIDTH
                        or height < MIN_PERSON_HEIGHT
                    ):
                        continue

                    frame_height, frame_width = frame.shape[:2]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_width, x2)
                    y2 = min(frame_height, y2)

                    person_roi = frame[y1:y2, x1:x2]

                    if run_pose:

                        pose_results = self.pose_estimator.estimate(
                            person_roi
                        )

                        self.pose_cache[person.track_id] = pose_results

                    else:

                        pose_results = self.pose_cache.get(
                            person.track_id
                        )

                active_ids = {
                    person.track_id
                    for person in tracked_people
                }

                self.pose_cache = {
                    track_id: pose
                    for track_id, pose in self.pose_cache.items()
                    if track_id in active_ids
                }

                Renderer.draw_tracked_people(
                    frame,
                    tracked_people,
                )

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