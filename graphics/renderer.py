"""
renderer.py

Provides drawing utilities for visualizing information on video frames.
"""

from __future__ import annotations

import cv2

from models.tracked_person import TrackedPerson
from models.person_detection import PersonDetection
from models.pose_result import PoseResult
from pose.pose_connections import POSE_CONNECTIONS

class Renderer:
    """
    Responsible for drawing overlays on video frames.
    """

    @staticmethod
    def draw_fps(frame: cv2.typing.MatLike, fps: float) -> None:
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


    @staticmethod
    def draw_detections(frame: cv2.typing.MatLike, detections: list[PersonDetection]) -> None:
        """
        Draw person detections on the frame.

        Args:
            frame: Video frame.
            detections: List of person detections.
        """

        for detection in detections:

            bbox = detection.bbox
            x1, y1, x2, y2 = bbox
            
            confidence = detection.confidence

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = f"Person {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )


    @staticmethod
    def draw_tracked_people(frame: cv2.typing.MatLike, tracked_people: list[TrackedPerson]) -> None:
        """
        Draw tracked people with their tracking IDs.

        Args:
            frame: Video frame.
            tracked_people: List of tracked people.
        """

        for person in tracked_people:

            bbox = person.bbox

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            Renderer.draw_pose(frame, person.pose)

            label = (
                f"ID: {person.track_id} | "
                f"Person {person.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )


    @staticmethod
    def draw_pose(frame: cv2.typing.MatLike, pose: PoseResult | None) -> None:
        """
        Draw pose landmarks and skeleton.
        """

        if pose is None:
            return
        
        landmarks = pose.landmarks

        if len(landmarks) == 0:
            return
        
        for start_index, end_index in POSE_CONNECTIONS:

            start = landmarks[start_index]
            end = landmarks[end_index]

            if (
                start.visibility < 0.5
                or end.visibility < 0.5
            ):
                continue

            cv2.line(
                frame,
                (start.x, start.y),
                (end.x, end.y),
                (0, 255, 255),
                2,
            )

        for landmark in landmarks:

            if landmark.visibility < 0.5:
                continue

            cv2.circle(
                frame,
                (landmark.x, landmark.y),
                4,
                (0, 255, 255),
                -1,
            )