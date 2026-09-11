"""Explainable, time-aware interaction risk scoring.

This is a research heuristic. Its output is a cue score, not a calibrated
probability or a determination that a person intends harm.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from analysis.pose_analyzer import PoseAnalyzer
from config.settings import (
    APPROACH_SPEED_FULL,
    APPROACH_SPEED_MAX_SCORE,
    APPROACH_SPEED_START,
    ARM_EXTENSION_FULL_RATIO,
    ARM_EXTENSION_HAND_SPEED_GATE,
    ARM_EXTENSION_MAX_SCORE,
    ARM_EXTENSION_START_RATIO,
    BODY_MOVEMENT_MAX_SCORE,
    BODY_MOVEMENT_MAX_SPEED,
    BODY_MOVEMENT_MIN_SPEED,
    HAND_SPEED_FULL_RISK,
    HAND_SPEED_MAX_SCORE,
    HAND_SPEED_MIN_SPEED,
    INTERACTION_DISTANCE_RATIO,
    INTERACTION_DURATION_FULL,
    INTERACTION_DURATION_MAX_SCORE,
    INTERACTION_DURATION_START,
    MAX_DELTA_SECONDS,
    MULTI_CUE_BONUS,
    POSE_MIN_VISIBILITY,
    PROXIMITY_FULL_RISK_RATIO,
    PROXIMITY_MAX_SCORE,
    PROXIMITY_START_RATIO,
    RISK_FALL_TAU_SECONDS,
    RISK_RISE_TAU_SECONDS,
    STATE_TTL_SECONDS,
)
from models.pose_landmark import PoseLandmark
from models.risk_breakdown import RiskBreakdown
from models.tracked_person import TrackedPerson
from models.types import RiskComputationResult, RiskScores, TrackPair


@dataclass(slots=True)
class _PersonSample:
    timestamp: float
    center: tuple[float, float]
    height: float
    wrists: tuple[PoseLandmark | None, PoseLandmark | None]


class RiskEngine:
    """Fuse normalized behavioral cues into an explainable temporal score."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._person_samples: dict[int, _PersonSample] = {}
        self._pair_samples: dict[TrackPair, tuple[float, float]] = {}
        self._pair_close_since: dict[TrackPair, float] = {}
        self._previous_risk_scores: RiskScores = {}
        self._last_seen: dict[int, float] = {}
        self.debug_scores: dict[int, RiskBreakdown] = {}

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(value, upper))

    @classmethod
    def _ramp(cls, value: float, start: float, full: float) -> float:
        if full <= start:
            return float(value >= full)
        return cls._clamp((value - start) / (full - start))

    @classmethod
    def _inverse_ramp(cls, value: float, start: float, full: float) -> float:
        if start <= full:
            return float(value <= full)
        return cls._clamp((start - value) / (start - full))

    @staticmethod
    def _pair_key(first_id: int, second_id: int) -> TrackPair:
        return min(first_id, second_id), max(first_id, second_id)

    @staticmethod
    def _height(person: TrackedPerson) -> float:
        return float(max(1, person.bbox[3] - person.bbox[1]))

    @staticmethod
    def _center(person: TrackedPerson) -> tuple[float, float]:
        if person.pose is not None:
            center = PoseAnalyzer.body_center(person.pose)
            if center is not None and center.visibility >= POSE_MIN_VISIBILITY:
                return float(center.x), float(center.y)
        x1, y1, x2, y2 = person.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _wrists(person: TrackedPerson) -> tuple[PoseLandmark | None, PoseLandmark | None]:
        if person.pose is None:
            return None, None
        wrists = PoseAnalyzer.wrists(person.pose)
        return tuple(
            point if point is not None and point.visibility >= POSE_MIN_VISIBILITY else None
            for point in wrists
        )  # type: ignore[return-value]

    @staticmethod
    def _pose_confidence(person: TrackedPerson) -> float:
        if person.pose is None or not person.pose.landmarks:
            return 0.0
        visible = [landmark.visibility for landmark in person.pose.landmarks]
        return sum(visible) / len(visible)

    def _arm_extension(self, person: TrackedPerson) -> float:
        if person.pose is None:
            return 0.0
        highest = 0.0
        for left in (True, False):
            shoulder, elbow, wrist = PoseAnalyzer.arm_landmarks(person.pose, left)
            if None in (shoulder, elbow, wrist):
                continue
            assert shoulder is not None and elbow is not None and wrist is not None
            if min(shoulder.visibility, elbow.visibility, wrist.visibility) < POSE_MIN_VISIBILITY:
                continue
            upper = math.hypot(elbow.x - shoulder.x, elbow.y - shoulder.y)
            lower = math.hypot(wrist.x - elbow.x, wrist.y - elbow.y)
            length = upper + lower
            if length > 0:
                direct = math.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y)
                highest = max(highest, direct / length)
        return self._ramp(highest, ARM_EXTENSION_START_RATIO, ARM_EXTENSION_FULL_RATIO)

    def _cleanup(self, now: float, active_ids: set[int], active_pairs: set[TrackPair]) -> None:
        stale_ids = {
            track_id
            for track_id, seen_at in self._last_seen.items()
            if track_id not in active_ids and now - seen_at > STATE_TTL_SECONDS
        }
        for store in (self._person_samples, self._previous_risk_scores, self._last_seen):
            for track_id in stale_ids:
                store.pop(track_id, None)
        for pair in list(self._pair_samples):
            if pair not in active_pairs and any(track_id in stale_ids for track_id in pair):
                self._pair_samples.pop(pair, None)
                self._pair_close_since.pop(pair, None)
        for pair in list(self._pair_close_since):
            if pair not in active_pairs:
                self._pair_close_since.pop(pair, None)

    def compute(
        self,
        people: list[TrackedPerson],
        timestamp: float | None = None,
    ) -> RiskComputationResult:
        """Return cue scores and an explanation for every active track."""
        now = self._clock() if timestamp is None else timestamp
        active_ids = {person.track_id for person in people}
        breakdowns = {person.track_id: RiskBreakdown() for person in people}
        current_samples: dict[int, _PersonSample] = {}

        for person in people:
            track_id = person.track_id
            height = self._height(person)
            center = self._center(person)
            wrists = self._wrists(person)
            breakdown = breakdowns[track_id]
            breakdown.pose_confidence = self._pose_confidence(person)
            previous = self._person_samples.get(track_id)
            current_samples[track_id] = _PersonSample(now, center, height, wrists)
            self._last_seen[track_id] = now
            if previous is None:
                continue
            dt = self._clamp(now - previous.timestamp, 1e-3, MAX_DELTA_SECONDS)
            scale = max(1.0, (height + previous.height) / 2.0)
            body_speed = math.dist(center, previous.center) / scale / dt
            breakdown.movement = (
                self._ramp(body_speed, BODY_MOVEMENT_MIN_SPEED, BODY_MOVEMENT_MAX_SPEED)
                * BODY_MOVEMENT_MAX_SCORE
            )
            hand_speeds: list[float] = []
            for current_wrist, previous_wrist in zip(wrists, previous.wrists):
                if current_wrist is not None and previous_wrist is not None:
                    hand_speeds.append(
                        math.hypot(
                            current_wrist.x - previous_wrist.x,
                            current_wrist.y - previous_wrist.y,
                        )
                        / scale
                        / dt
                    )
            if hand_speeds:
                speed = max(hand_speeds)
                breakdown.hand_speed = (
                    self._ramp(speed, HAND_SPEED_MIN_SPEED, HAND_SPEED_FULL_RISK)
                    * HAND_SPEED_MAX_SCORE
                )
            extension = self._arm_extension(person)
            if breakdown.hand_speed >= ARM_EXTENSION_HAND_SPEED_GATE:
                breakdown.arm_extension = extension * ARM_EXTENSION_MAX_SCORE

        active_pairs: set[TrackPair] = set()
        for index, first in enumerate(people):
            for second in people[index + 1 :]:
                pair = self._pair_key(first.track_id, second.track_id)
                active_pairs.add(pair)
                first_sample = current_samples[first.track_id]
                second_sample = current_samples[second.track_id]
                average_height = max(1.0, (first_sample.height + second_sample.height) / 2.0)
                distance_ratio = math.dist(first_sample.center, second_sample.center) / average_height
                proximity = (
                    self._inverse_ramp(
                        distance_ratio,
                        PROXIMITY_START_RATIO,
                        PROXIMITY_FULL_RISK_RATIO,
                    )
                    * PROXIMITY_MAX_SCORE
                )
                for person in (first, second):
                    breakdowns[person.track_id].proximity = max(
                        breakdowns[person.track_id].proximity, proximity
                    )

                previous_pair = self._pair_samples.get(pair)
                if previous_pair is not None:
                    previous_ratio, previous_time = previous_pair
                    dt = self._clamp(now - previous_time, 1e-3, MAX_DELTA_SECONDS)
                    closing_speed = (previous_ratio - distance_ratio) / dt
                    approach = (
                        self._ramp(closing_speed, APPROACH_SPEED_START, APPROACH_SPEED_FULL)
                        * APPROACH_SPEED_MAX_SCORE
                    )
                    for person in (first, second):
                        breakdowns[person.track_id].approach_speed = max(
                            breakdowns[person.track_id].approach_speed, approach
                        )
                self._pair_samples[pair] = distance_ratio, now

                if distance_ratio <= INTERACTION_DISTANCE_RATIO:
                    close_since = self._pair_close_since.setdefault(pair, now)
                    duration = now - close_since
                    duration_score = (
                        self._ramp(
                            duration,
                            INTERACTION_DURATION_START,
                            INTERACTION_DURATION_FULL,
                        )
                        * INTERACTION_DURATION_MAX_SCORE
                    )
                    for person in (first, second):
                        breakdowns[person.track_id].interaction_duration = max(
                            breakdowns[person.track_id].interaction_duration,
                            duration_score,
                        )
                else:
                    self._pair_close_since.pop(pair, None)

        risk_scores: RiskScores = {}
        for person in people:
            track_id = person.track_id
            breakdown = breakdowns[track_id]
            contributions = breakdown.contributions()
            active_cues = sum(value >= 0.025 for value in contributions.values())
            if active_cues >= 3 and (breakdown.proximity > 0 or breakdown.approach_speed > 0):
                breakdown.multi_cue_bonus = min(MULTI_CUE_BONUS, (active_cues - 2) * 0.035)
            raw = self._clamp(sum(breakdown.contributions().values()))
            breakdown.raw_total = raw
            previous_risk = self._previous_risk_scores.get(track_id)
            previous_sample = self._person_samples.get(track_id)
            if previous_risk is None or previous_sample is None:
                smoothed = raw
            else:
                dt = self._clamp(now - previous_sample.timestamp, 1e-3, MAX_DELTA_SECONDS)
                tau = RISK_RISE_TAU_SECONDS if raw > previous_risk else RISK_FALL_TAU_SECONDS
                alpha = 1.0 - math.exp(-dt / tau)
                smoothed = previous_risk + alpha * (raw - previous_risk)
            smoothed = self._clamp(smoothed)
            breakdown.smoothed_total = smoothed
            risk_scores[track_id] = smoothed
            self._previous_risk_scores[track_id] = smoothed

        self._person_samples.update(current_samples)
        self._cleanup(now, active_ids, active_pairs)
        self.debug_scores = breakdowns
        return risk_scores, breakdowns
