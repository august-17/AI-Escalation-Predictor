"""
Application controller for the AI Escalation Predictor.
"""

from __future__ import annotations

import logging

import cv2

import time

from camera.camera import Camera
from camera.fps import FPSCounter
from graphics.renderer import Renderer
from tracking.person_tracker import PersonTracker
from pose.landmark_converter import LandmarkConverter
from models.pose_result import PoseResult
from pose.pose_estimator import PoseEstimator
from analysis.risk_engine import RiskEngine
from config.settings import (
    POSE_INTERVAL_SINGLE_PERSON,
    POSE_INTERVAL_FEW_PEOPLE,
    POSE_INTERVAL_MANY_PEOPLE,
    MIN_PERSON_WIDTH,
    MIN_PERSON_HEIGHT,
    POSE_CACHE_TIMEOUT
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

        self.frame_count: int = 0
        self.pose_cache: dict[int, tuple[PoseResult, float]] = {}

        self.risk_engine = RiskEngine()
    

    def _should_run_pose(self, people_count: int) -> bool:
        """
        Determine whether pose estimation should run
        for the current frame.
        """

        if people_count == 0:
            return False

        if people_count == 1:
            return (self.frame_count % POSE_INTERVAL_SINGLE_PERSON == 0)

        if people_count <= 3:
            return (self.frame_count % POSE_INTERVAL_FEW_PEOPLE == 0)

        return (self.frame_count % POSE_INTERVAL_MANY_PEOPLE == 0)
    

    def _extract_person_roi(self, frame: cv2.typing.MatLike, person) -> cv2.typing.MatLike | None:
        """
        Extract a valid person ROI from the frame.

        Args:
            frame: Current video frame.
            person: Tracked person.

        Returns:
            Cropped person ROI, or None if the ROI is too small.
        """

        x1, y1, x2, y2 = person.bbox

        width = x2 - x1
        height = y2 - y1

        if (
            width < MIN_PERSON_WIDTH
            or height < MIN_PERSON_HEIGHT
        ):
            return None

        frame_height, frame_width = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        return frame[y1:y2, x1:x2]


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

                start = time.perf_counter()
                tracked_people = self.tracker.track(frame)
                yolo_time = time.perf_counter() - start

                self.frame_count += 1

                run_pose = self._should_run_pose(len(tracked_people))

                pose_start = time.perf_counter()

                current_time = time.perf_counter()

                for person in tracked_people:

                    person_roi = self._extract_person_roi(frame, person)

                    if person_roi is None:
                        continue

                    if run_pose:

                        mp_pose_result = self.pose_estimator.estimate(person.track_id, person_roi)

                        x1, y1, x2, y2 = person.bbox

                        pose_result = LandmarkConverter.convert(
                            mp_pose_result,
                            roi_x=x1,
                            roi_y=y1,
                            roi_width=x2 - x1,
                            roi_height=y2 - y1
                        )

                        if len(pose_result.landmarks) == 33:

                            self.pose_cache[person.track_id] = (pose_result, current_time)

                        else:

                            cached_pose = self.pose_cache.get(person.track_id)

                            if cached_pose is not None:

                                pose_result, timestamp = cached_pose

                                if (current_time - timestamp > POSE_CACHE_TIMEOUT):

                                    pose_result = None

                            else:

                                pose_result = None
                    else:

                        cached_pose = self.pose_cache.get(person.track_id)

                        if cached_pose is not None:

                            pose_result, timestamp = cached_pose

                            if (current_time - timestamp > POSE_CACHE_TIMEOUT):

                                pose_result = None

                        else:

                            pose_result = None

                    person.pose = pose_result

                pose_time = time.perf_counter() - pose_start

                active_ids = {
                    person.track_id
                    for person in tracked_people
                }

                self.pose_cache = {
                    track_id: pose
                    for track_id, pose in self.pose_cache.items()
                    if track_id in active_ids
                }

                self.pose_estimator.remove_inactive(active_ids)
                
                start = time.perf_counter()
                risk_scores, debug_scores = self.risk_engine.compute(tracked_people)
                risk_time = time.perf_counter() - start

                if self.frame_count % 15 == 0:
                    
                    print()

                    for person in tracked_people:

                        scores = debug_scores[person.track_id]

                        print(f"ID {person.track_id}")

                        print(
                            f"  Proximity      : "
                            f"{scores.proximity:.3f}"
                        )

                        print(
                            f"  Movement       : "
                            f"{scores.movement:.3f}"
                        )

                        print(
                            f"  Raw Total      : "
                            f"{scores.raw_total:.3f}"
                        )

                        print(
                            f"  Smoothed Total : "
                            f"{scores.smoothed_total:.3f}"
                        )

                        print()

                if self.frame_count % 60 == 0:
                    logger.info(
                        "YOLO: %.3f ms | Pose: %.3f ms | Risk: %.3f ms",
                        yolo_time * 1000,
                        pose_time * 1000,
                        risk_time * 1000,
                    )

                Renderer.draw_tracked_people(frame, tracked_people, risk_scores)

                for person in tracked_people:

                    Renderer.draw_pose(frame, person.pose)

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