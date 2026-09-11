"""Configurable webcam/video application controller."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2

from alerts.alert_event import AlertEvent
from alerts.alert_manager import AlertManager
from alerts.event_logger import EventLogger
from analysis.risk_engine import RiskEngine
from camera.camera import Camera
from camera.fps import FPSCounter
from config.runtime import RuntimeConfig
from config.settings import (
    MIN_PERSON_HEIGHT,
    MIN_PERSON_WIDTH,
    POSE_CACHE_TIMEOUT,
    POSE_INTERVAL_FEW_PEOPLE,
    POSE_INTERVAL_MANY_PEOPLE,
    POSE_INTERVAL_SINGLE_PERSON,
    WINDOW_NAME,
)
from graphics.renderer import Renderer
from models.pose_result import PoseResult
from pose.landmark_converter import LandmarkConverter

logger = logging.getLogger(__name__)


class Application:
    """Coordinate capture, tracking, pose analysis, scoring, alerts and output."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        camera: Camera | None = None,
        tracker: Any | None = None,
        pose_estimator: Any | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.camera = camera or Camera(self.config.source, self.config.width, self.config.height)
        if tracker is None:
            from tracking.person_tracker import PersonTracker

            tracker = PersonTracker(self.config.model, self.config.device)
        self.tracker = tracker
        if self.config.disable_pose:
            self.pose_estimator = None
        elif pose_estimator is not None:
            self.pose_estimator = pose_estimator
        else:
            from pose.pose_estimator import PoseEstimator

            self.pose_estimator = PoseEstimator()
        self.fps_counter = FPSCounter()
        self.risk_engine = RiskEngine()
        self.alert_manager = AlertManager()
        self.event_logger = EventLogger(self.config.events_log)
        self.frame_count = 0
        self.pose_cache: dict[int, tuple[PoseResult, float]] = {}
        self.last_performance_log = time.perf_counter()
        self.writer: cv2.VideoWriter | None = None

    def _should_run_pose(self, people_count: int) -> bool:
        if self.pose_estimator is None or people_count == 0:
            return False
        interval = (
            POSE_INTERVAL_SINGLE_PERSON
            if people_count == 1
            else POSE_INTERVAL_FEW_PEOPLE
            if people_count <= 3
            else POSE_INTERVAL_MANY_PEOPLE
        )
        return self.frame_count % interval == 0

    @staticmethod
    def _extract_person_roi(frame: cv2.typing.MatLike, person: Any) -> cv2.typing.MatLike | None:
        x1, y1, x2, y2 = person.bbox
        frame_height, frame_width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        if x2 - x1 < MIN_PERSON_WIDTH or y2 - y1 < MIN_PERSON_HEIGHT:
            return None
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size else None

    def _attach_poses(self, frame: cv2.typing.MatLike, people: list[Any], now: float) -> str:
        run_pose = self._should_run_pose(len(people))
        status = "disabled" if self.pose_estimator is None else "estimated" if run_pose else "cached"
        for person in people:
            pose_result: PoseResult | None = None
            if run_pose:
                roi = self._extract_person_roi(frame, person)
                if roi is not None:
                    raw_pose = self.pose_estimator.estimate(person.track_id, roi)
                    x1, y1, x2, y2 = person.bbox
                    candidate = LandmarkConverter.convert(raw_pose, x1, y1, x2 - x1, y2 - y1)
                    if len(candidate.landmarks) == 33:
                        pose_result = candidate
                        self.pose_cache[person.track_id] = candidate, now
            if pose_result is None:
                cached = self.pose_cache.get(person.track_id)
                if cached is not None and now - cached[1] <= POSE_CACHE_TIMEOUT:
                    pose_result = cached[0]
            person.pose = pose_result
        active_ids = {person.track_id for person in people}
        self.pose_cache = {
            track_id: value for track_id, value in self.pose_cache.items() if track_id in active_ids
        }
        if self.pose_estimator is not None:
            self.pose_estimator.remove_inactive(active_ids)
        return status

    def _handle_alerts(self, events: list[AlertEvent]) -> None:
        for event in events:
            self.event_logger.write(event)
            logger.warning(
                "[CONFIRMED %s] track=%d score=%.2f primary_cue=%s",
                event.level.name,
                event.track_id,
                event.risk,
                event.dominant_factor,
            )

    def process_frame(
        self,
        frame: cv2.typing.MatLike,
        timestamp: float,
    ) -> cv2.typing.MatLike:
        started = time.perf_counter()
        people = self.tracker.track(frame)
        tracking_ms = (time.perf_counter() - started) * 1000
        self.frame_count += 1
        pose_started = time.perf_counter()
        pose_status = self._attach_poses(frame, people, timestamp)
        pose_ms = (time.perf_counter() - pose_started) * 1000
        scores, breakdowns = self.risk_engine.compute(people, timestamp)
        states = self.alert_manager.update(scores, breakdowns, timestamp)
        self._handle_alerts(self.alert_manager.get_new_events())
        self.fps_counter.update()
        fps = self.fps_counter.get_fps()
        Renderer.render(frame, people, scores, states, breakdowns, fps)
        now = time.perf_counter()
        if now - self.last_performance_log >= 5.0:
            logger.info(
                "fps=%.1f tracks=%d pose=%s pose_ms=%.1f tracking_ms=%.1f",
                fps,
                len(people),
                pose_status,
                pose_ms,
                tracking_ms,
            )
            self.last_performance_log = now
        return frame

    def _open_writer(self, frame: cv2.typing.MatLike, fps: float) -> None:
        if self.config.output is None or self.writer is not None:
            return
        target = Path(self.config.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self.writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 25.0, (width, height)
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"Unable to create output video: {target}")

    def run(self) -> int:
        logger.info("Starting research prototype with source %s", self.config.source)
        if not self.camera.open():
            return 1
        source_fps = self.camera.get_fps()
        try:
            while self.config.max_frames is None or self.frame_count < self.config.max_frames:
                success, frame = self.camera.read()
                if not success or frame is None:
                    if self.camera.is_file:
                        logger.info("Reached end of video source.")
                    else:
                        logger.error("Unable to read webcam frame.")
                    break
                timestamp = (
                    self.frame_count / source_fps
                    if self.camera.is_file and source_fps > 0
                    else time.perf_counter()
                )
                annotated = self.process_frame(frame, timestamp)
                self._open_writer(annotated, source_fps)
                if self.writer is not None:
                    self.writer.write(annotated)
                if not self.config.headless:
                    cv2.imshow(WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break
            return 0
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            return 130
        except Exception:
            logger.exception("Application failed.")
            return 2
        finally:
            self.camera.release()
            if self.writer is not None:
                self.writer.release()
            if self.pose_estimator is not None:
                self.pose_estimator.close()
            if not self.config.headless:
                cv2.destroyAllWindows()
