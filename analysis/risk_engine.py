"""
Computes an escalation risk score for tracked people.
"""

from __future__ import annotations

import math

import logging

from analysis.pose_analyzer import PoseAnalyzer
from models.tracked_person import TrackedPerson
from models.risk_breakdown import RiskBreakdown
from models.pose_result import PoseResult
from models.pose_landmark import PoseLandmark
from models.types import (
    TrackPair,
    RiskScores,
    RiskComputationResult
)
from config.settings import (
    PROXIMITY_START_DISTANCE,
    PROXIMITY_FULL_RISK_DISTANCE,
    PROXIMITY_RISK_WEIGHT,
    BODY_MOVEMENT_MIN_SPEED,
    BODY_MOVEMENT_MAX_SPEED,
    BODY_MOVEMENT_RISK_WEIGHT,
    RISK_INCREASE_MEMORY,
    RISK_DECREASE_MEMORY,
    HAND_SPEED_MAX_PIXELS,
    HAND_SPEED_FULL_RISK,
    HAND_SPEED_MIN_SPEED,
    HAND_SPEED_MAX_SCORE,
    ARM_EXTENSION_START_RATIO,
    ARM_EXTENSION_FULL_RATIO,
    ARM_EXTENSION_MAX_SCORE,
    ARM_EXTENSION_HAND_SPEED_GATE,
    APPROACH_SPEED_START,
    APPROACH_SPEED_FULL,
    APPROACH_SPEED_MAX_SCORE,
    ESCALATION_BUILD_RATE,
    ESCALATION_DECAY_RATE,
    ESCALATION_DECAY_THRESHOLD,
    ESCALATION_CONFIDENCE_WEIGHT
)

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Computes an escalation score.

    Current implementation is a placeholder that always
    returns zero until behavioral features are added.
    """

    def __init__(self) -> None:
        """
        Initialize the risk engine.
        """

        self._previous_body_centers: dict[int, PoseLandmark] = {}

        self._previous_pose: dict[int, PoseResult] = {}

        self._previous_distances: dict[TrackPair, float] = {}

        self._previous_risk_scores: RiskScores = {}

        self._escalation_confidence: RiskScores = {}

        self.debug_scores: dict[int, RiskBreakdown] = {}


    @staticmethod
    def _pair_key(
        first_id: int,
        second_id: int
    ) -> TrackPair:
        return (
            min(first_id, second_id),
            max(first_id, second_id)
        )


    @staticmethod
    def _distance_between_people(
        first: TrackedPerson,
        second: TrackedPerson
    ) -> float:
        """
        Compute the Euclidean distance between the body centers
        of two tracked people.
        """

        if first.pose is None or second.pose is None:
            return float("inf")

        first_center = PoseAnalyzer.body_center(first.pose)
        second_center = PoseAnalyzer.body_center(second.pose)

        if first_center is None or second_center is None:
            return float("inf")

        return math.hypot(
            first_center.x - second_center.x,
            first_center.y - second_center.y
        )
    

    def _apply_risk_memory(
        self,
        people: list[TrackedPerson],
        risk_scores: RiskScores
    ) -> None:
        """
        Smooth risk scores over time.
        """

        active_ids: set[int] = set()

        for person in people:

            active_ids.add(person.track_id)

            previous = self._previous_risk_scores.get(person.track_id, 0.0)

            current = risk_scores[person.track_id]

            if current > previous:

                smoothed = (
                    previous * RISK_INCREASE_MEMORY
                    + current * (1.0 - RISK_INCREASE_MEMORY)
                )

            else:

                smoothed = (
                    previous * RISK_DECREASE_MEMORY
                    + current * (1.0 - RISK_DECREASE_MEMORY)
                )

            smoothed = max(0.0, min(smoothed, 1.0))

            risk_scores[person.track_id] = smoothed

            self._previous_risk_scores[person.track_id] = smoothed

        self._previous_risk_scores = {
            track_id: risk
            for track_id, risk
            in self._previous_risk_scores.items()
            if track_id in active_ids
        }


    def _apply_escalation_persistence(
        self,
        people: list[TrackedPerson],
        risk_scores: RiskScores
    ) -> None:
        """
        Maintain long-term escalation confidence
        for each tracked person.
        """

        active_ids: set[int] = set()

        for person in people:

            active_ids.add(person.track_id)

            confidence = self._escalation_confidence.get(person.track_id, 0.0)

            risk = risk_scores[person.track_id]

            if risk < ESCALATION_DECAY_THRESHOLD:

                confidence -= ESCALATION_DECAY_RATE

            else:

                confidence += risk * ESCALATION_BUILD_RATE

            confidence = max(0.0, min(confidence, 1.0))

            risk_scores[person.track_id] += confidence * ESCALATION_CONFIDENCE_WEIGHT

            risk_scores[person.track_id] = max(0.0, min(risk_scores[person.track_id], 1.0))

            self._escalation_confidence[person.track_id] = confidence

        self._escalation_confidence = {
            track_id: confidence
            for track_id, confidence
            in self._escalation_confidence.items()
            if track_id in active_ids
        }


    def _compute_proximity_risk(self, people: list[TrackedPerson]) -> RiskScores:
        """
        Update risk scores based on the proximity of tracked people.
        """

        proximity_scores = {
            person.track_id: 0.0
            for person in people
        }

        for i, first in enumerate(people):

            for second in people[i + 1:]:

                distance = self._distance_between_people(
                    first,
                    second
                )

                if distance >= PROXIMITY_START_DISTANCE:
                    continue

                normalized = (PROXIMITY_START_DISTANCE - distance) / (PROXIMITY_START_DISTANCE - PROXIMITY_FULL_RISK_DISTANCE)

                normalized = max(0.0, min(normalized, 1.0))

                proximity_risk = (normalized * PROXIMITY_RISK_WEIGHT)

                proximity_scores[first.track_id] += proximity_risk
                proximity_scores[second.track_id] += proximity_risk

        return proximity_scores


    def _compute_body_movement_risk(self, people: list[TrackedPerson]) -> RiskScores:
        """
        Update risk scores based on body movement.
        """
        movement_scores = {
            person.track_id: 0.0
            for person in people
        }
        active_ids: set[int] = set()

        for person in people:

            active_ids.add(person.track_id)

            if person.pose is None:
                continue

            center = PoseAnalyzer.body_center(person.pose)

            if center is None:
                continue

            current = (center.x, center.y)

            previous = self._previous_body_centers.get(person.track_id)

            if previous is not None:

                movement = math.hypot(
                    current[0] - previous[0],
                    current[1] - previous[1]
                )

                if movement > BODY_MOVEMENT_MIN_SPEED:

                    normalized = (movement - BODY_MOVEMENT_MIN_SPEED) / (BODY_MOVEMENT_MAX_SPEED - BODY_MOVEMENT_MIN_SPEED)

                    normalized = max(0.0, min(normalized, 1.0))

                    movement_scores[person.track_id] += (normalized * BODY_MOVEMENT_RISK_WEIGHT)

            self._previous_body_centers[person.track_id] = current

        self._previous_body_centers = {
            track_id: center
            for track_id, center
            in self._previous_body_centers.items()
            if track_id in active_ids
        }

        return movement_scores
    

    def _compute_hand_speed_risk(self, people: list[TrackedPerson]) -> RiskScores:
        """
        Compute risk based on wrist movement speed.
        """

        hand_speed_scores = {
            person.track_id: 0.0
            for person in people
        }

        for person in people:

            if person.pose is None:
                continue

            current_left, current_right = PoseAnalyzer.wrists(person.pose)

            if (
                current_left is None
                or current_right is None
            ):
                continue

            previous_pose = self._previous_pose.get(person.track_id)

            if previous_pose is None:

                self._previous_pose[person.track_id] = person.pose

                continue

            previous_left, previous_right = PoseAnalyzer.wrists(previous_pose)

            if (
                previous_left is None
                or previous_right is None
            ):

                self._previous_pose[person.track_id] = person.pose

                continue

            left_speed = math.hypot(
                current_left.x - previous_left.x,
                current_left.y - previous_left.y,
            )

            right_speed = math.hypot(
                current_right.x - previous_right.x,
                current_right.y - previous_right.y,
            )

            hand_speed = (left_speed + right_speed) / 2

            hand_speed = min(hand_speed, HAND_SPEED_MAX_PIXELS)

            if hand_speed > HAND_SPEED_MIN_SPEED:

                risk = (hand_speed - HAND_SPEED_MIN_SPEED) / (HAND_SPEED_FULL_RISK - HAND_SPEED_MIN_SPEED)

                risk = max(0.0, min(risk, 1.0))

                hand_speed_scores[person.track_id] = (risk * HAND_SPEED_MAX_SCORE)

                logger.debug(
                    "ID %d | Hand Speed: %.1f px | Hand Risk: %.3f",
                    person.track_id,
                    hand_speed,
                    risk,
                )

            self._previous_pose[person.track_id] = person.pose

        return hand_speed_scores
    

    def _compute_arm_extension_risk(self, people: list[TrackedPerson]) -> RiskScores:
        """
        Compute risk based on arm extension.
        """

        arm_extension_scores = {
            person.track_id: 0.0
            for person in people
        }

        for person in people:

            if person.pose is None:
                continue

            highest_ratio = 0.0

            for left in (True, False):

                shoulder, elbow, wrist = PoseAnalyzer.arm_landmarks(person.pose, left)

                if (shoulder is None or elbow is None or wrist is None):
                    continue

                shoulder_to_elbow = math.hypot(
                    elbow.x - shoulder.x,
                    elbow.y - shoulder.y
                )

                elbow_to_wrist = math.hypot(
                    wrist.x - elbow.x,
                    wrist.y - elbow.y
                )

                shoulder_to_wrist = math.hypot(
                    wrist.x - shoulder.x,
                    wrist.y - shoulder.y
                )

                arm_length = shoulder_to_elbow + elbow_to_wrist

                if arm_length <= 0:
                    continue

                extension_ratio = shoulder_to_wrist / arm_length

                highest_ratio = max(highest_ratio, extension_ratio)

            if highest_ratio > ARM_EXTENSION_START_RATIO:

                risk = (highest_ratio - ARM_EXTENSION_START_RATIO) / (ARM_EXTENSION_FULL_RATIO - ARM_EXTENSION_START_RATIO)

                risk = max(0.0, min(risk, 1.0))

                arm_extension_scores[person.track_id] = risk * ARM_EXTENSION_MAX_SCORE

        return arm_extension_scores
    

    def _compute_approach_speed_risk(self, people: list[TrackedPerson]) -> RiskScores:
        """
        Compute risk based on how quickly
        two people move toward each other.
        """

        approach_scores = {
            person.track_id: 0.0
            for person in people
        }

        active_pairs: set[TrackPair] = set()

        for i, first in enumerate(people):

            for second in people[i + 1:]:

                distance = self._distance_between_people(first, second)

                if distance > PROXIMITY_START_DISTANCE:
                    continue

                pair = self._pair_key(
                    first.track_id,
                    second.track_id
                )

                active_pairs.add(pair)

                previous_distance = self._previous_distances.get(pair)

                if previous_distance is None:

                    self._previous_distances[pair] = distance

                    continue

                closing_speed = previous_distance - distance

                if closing_speed <= 0:

                    self._previous_distances[pair] = distance

                    continue

                normalized = (closing_speed - APPROACH_SPEED_START) / (APPROACH_SPEED_FULL - APPROACH_SPEED_START)

                normalized = max(0.0, min(normalized, 1.0))

                approach_risk = normalized * APPROACH_SPEED_MAX_SCORE

                approach_scores[first.track_id] += approach_risk

                approach_scores[second.track_id] += approach_risk

                logger.debug(
                    "Pair %s | Closing Speed: %.1f px | Approach Risk: %.3f",
                    pair,
                    closing_speed,
                    approach_risk,
                )

                self._previous_distances[pair] = distance

        self._previous_distances = {
                    pair: distance
                    for pair, distance
                    in self._previous_distances.items()
                    if pair in active_pairs
                }

        return approach_scores
    

    def _combine_risk_scores(
        self,
        people: list[TrackedPerson],
        proximity_scores: RiskScores,
        movement_scores: RiskScores,
        hand_speed_scores: RiskScores,
        arm_extension_scores: RiskScores,
        approach_speed_scores: RiskScores
    ) -> RiskComputationResult:
        """
        Combine individual feature scores into a single
        risk score for each tracked person.
        """

        risk_scores: RiskScores = {}

        debug_scores: dict[int, RiskBreakdown] = {}

        for person in people:

            proximity = proximity_scores.get(person.track_id, 0.0)

            movement = movement_scores.get(person.track_id, 0.0)

            hand_speed = hand_speed_scores.get(person.track_id, 0.0)

            arm_extension = arm_extension_scores.get(person.track_id, 0.0)

            approach_speed = approach_speed_scores.get(person.track_id, 0.0)

            if hand_speed < ARM_EXTENSION_HAND_SPEED_GATE:
                arm_extension = 0.0

            total = proximity + movement + hand_speed + arm_extension + approach_speed

            risk_scores[person.track_id] = total

            debug_scores[person.track_id] = RiskBreakdown(
                proximity=proximity,
                movement=movement,
                hand_speed=hand_speed,
                arm_extension=arm_extension,
                approach_speed=approach_speed,
                raw_total=total
            )

        return (risk_scores, debug_scores)


    def compute(self, people: list[TrackedPerson]) -> RiskComputationResult:
        """
        Compute a risk score for each tracked person.

        Args:
            people: List of tracked people.

        Returns:
            Dictionary mapping track IDs to risk scores.
        """

        proximity_scores = self._compute_proximity_risk(people)

        movement_scores = self._compute_body_movement_risk(people)

        hand_speed_scores = self._compute_hand_speed_risk(people)

        arm_extension_scores = self._compute_arm_extension_risk(people)

        approach_speed_scores = self._compute_approach_speed_risk(people)

        risk_scores, debug_scores = (
            self._combine_risk_scores(
                people,
                proximity_scores,
                movement_scores,
                hand_speed_scores,
                arm_extension_scores,
                approach_speed_scores
            )
        )

        self._apply_risk_memory(people, risk_scores)

        self._apply_escalation_persistence(people, risk_scores)

        for person in people:

            debug_scores[person.track_id].smoothed_total = risk_scores[person.track_id]

        return (risk_scores, debug_scores)