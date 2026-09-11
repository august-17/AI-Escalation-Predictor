"""OpenCV overlays for tracks, poses, cue explanations and system status."""

from __future__ import annotations

import cv2

from alerts.alert_level import AlertLevel
from alerts.alert_state import AlertState
from config.settings import POSE_MIN_VISIBILITY, WINDOW_NAME
from models.pose_result import PoseResult
from models.risk_breakdown import RiskBreakdown
from models.tracked_person import TrackedPerson
from models.types import RiskScores
from pose.pose_connections import POSE_CONNECTIONS

_COLORS = {
    AlertLevel.NORMAL: (104, 211, 145),
    AlertLevel.WATCH: (77, 210, 240),
    AlertLevel.WARNING: (60, 153, 255),
    AlertLevel.CRITICAL: (82, 82, 245),
}


class Renderer:
    @staticmethod
    def draw_pose(frame: cv2.typing.MatLike, pose: PoseResult | None) -> None:
        if pose is None or len(pose.landmarks) != 33:
            return
        for start_index, end_index in POSE_CONNECTIONS:
            start, end = pose.landmarks[start_index], pose.landmarks[end_index]
            if min(start.visibility, end.visibility) < POSE_MIN_VISIBILITY:
                continue
            cv2.line(frame, (start.x, start.y), (end.x, end.y), (230, 205, 95), 2)
        for landmark in pose.landmarks:
            if landmark.visibility >= POSE_MIN_VISIBILITY:
                cv2.circle(frame, (landmark.x, landmark.y), 3, (240, 235, 165), -1)

    @staticmethod
    def draw_tracks(
        frame: cv2.typing.MatLike,
        people: list[TrackedPerson],
        scores: RiskScores,
        states: dict[int, AlertState],
        breakdowns: dict[int, RiskBreakdown],
    ) -> None:
        for person in people:
            x1, y1, x2, y2 = person.bbox
            state = states.get(person.track_id)
            level = state.level if state else AlertLevel.NORMAL
            color = _COLORS[level]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            Renderer.draw_pose(frame, person.pose)
            label = f"ID {person.track_id}  cue {scores.get(person.track_id, 0.0):.2f}  {level.name}"
            label_y = max(24, y1 - 10)
            (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 1)
            cv2.rectangle(frame, (x1, label_y - 20), (x1 + text_w + 10, label_y + 5), (13, 22, 34), -1)
            cv2.putText(frame, label, (x1 + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_header(frame: cv2.typing.MatLike, fps: float, people_count: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (8, 16, 28), -1)
        cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
        cv2.putText(frame, "AI ESCALATION PREDICTOR", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (225, 238, 250), 1, cv2.LINE_AA)
        status = f"RESEARCH PROTOTYPE   FPS {fps:.1f}   TRACKS {people_count}"
        (width, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(frame, status, (max(18, frame.shape[1] - width - 18), 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (157, 191, 219), 1, cv2.LINE_AA)

    @staticmethod
    def draw_summary(
        frame: cv2.typing.MatLike,
        scores: RiskScores,
        states: dict[int, AlertState],
        breakdowns: dict[int, RiskBreakdown],
    ) -> None:
        if not scores:
            return
        track_id, risk = max(scores.items(), key=lambda item: item[1])
        level = states.get(track_id, AlertState(track_id)).level
        color = _COLORS[level]
        factor = breakdowns.get(track_id, RiskBreakdown()).dominant_factor.replace("_", " ")
        text = f"Highest cue: {risk:.2f}  |  Track {track_id}  |  {level.value}  |  Primary: {factor}"
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 46), (frame.shape[1], frame.shape[0]), (8, 16, 28), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.putText(frame, text, (18, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)

    @staticmethod
    def render(
        frame: cv2.typing.MatLike,
        people: list[TrackedPerson],
        scores: RiskScores,
        states: dict[int, AlertState],
        breakdowns: dict[int, RiskBreakdown],
        fps: float,
    ) -> cv2.typing.MatLike:
        Renderer.draw_tracks(frame, people, scores, states, breakdowns)
        Renderer.draw_header(frame, fps, len(people))
        Renderer.draw_summary(frame, scores, states, breakdowns)
        return frame
