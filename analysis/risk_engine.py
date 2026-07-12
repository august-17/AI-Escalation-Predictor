"""
Computes an escalation risk score for tracked people.
"""

from __future__ import annotations

import math

from analysis.pose_analyzer import PoseAnalyzer
from models.tracked_person import TrackedPerson
from models.risk_breakdown import RiskBreakdown
from models.pose_result import PoseResult
from models.pose_landmark import PoseLandmark
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
    ARM_EXTENSION_HAND_SPEED_GATE
)


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

        self.previous_pose: dict[int, PoseResult] = {}

        self._previous_risk_scores: dict[int, float] = {}

        self.debug_scores: dict[int, RiskBreakdown] = {}


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


    def _compute_proximity_risk(
        self,
        people: list[TrackedPerson]
    ) -> dict[int, float]:
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


    def _compute_body_movement_risk(
        self,
        people: list[TrackedPerson]
    ) -> dict[int, float]:
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

            previous = self._previous_body_centers.get(
                person.track_id
            )

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


    def _apply_risk_memory(
        self,
        people: list[TrackedPerson],
        risk_scores: dict[int, float]
    ) -> None:
        """
        Smooth risk scores over time.
        """

        active_ids: set[int] = set()

        for person in people:

            active_ids.add(person.track_id)

            previous = self._previous_risk_scores.get(
                person.track_id,
                0.0
            )

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

            self._previous_risk_scores[
                person.track_id
            ] = smoothed

        self._previous_risk_scores = {
            track_id: risk
            for track_id, risk
            in self._previous_risk_scores.items()
            if track_id in active_ids
        }


    def _combine_risk_scores(
        self,
        people: list[TrackedPerson],
        proximity_scores: dict[int, float],
        movement_scores: dict[int, float],
        hand_speed_scores: dict[int, float],
        arm_extension_scores: dict[int, float]
    ) -> tuple[
        dict[int, float],
        dict[int, RiskBreakdown]
    ]:
        """
        Combine individual feature scores into a single
        risk score for each tracked person.
        """

        risk_scores: dict[int, float] = {}

        debug_scores: dict[int, RiskBreakdown] = {}

        for person in people:

            proximity = proximity_scores.get(person.track_id, 0.0)

            movement = movement_scores.get(person.track_id, 0.0)

            hand_speed = hand_speed_scores.get(person.track_id, 0.0)

            arm_extension = arm_extension_scores.get(person.track_id, 0.0)

            if hand_speed < ARM_EXTENSION_HAND_SPEED_GATE:
                arm_extension = 0.0

            total = proximity + movement + hand_speed + arm_extension

            risk_scores[person.track_id] = total

            debug_scores[person.track_id] = RiskBreakdown(
                proximity=proximity,
                movement=movement,
                hand_speed=hand_speed,
                arm_extension=arm_extension,
                raw_total=total
            )

        return (risk_scores, debug_scores)


    def compute(
        self,
        people: list[TrackedPerson]
    ) -> tuple[
        dict[int, float],
        dict[int, RiskBreakdown]
    ]:
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

        risk_scores, debug_scores = (
            self._combine_risk_scores(
                people,
                proximity_scores,
                movement_scores,
                hand_speed_scores,
                arm_extension_scores
            )
        )

        self._apply_risk_memory(people, risk_scores)

        for person in people:

            debug_scores[person.track_id].smoothed_total = risk_scores[person.track_id]

        return (risk_scores, debug_scores)
    

    def _compute_hand_speed_risk(self, people: list[TrackedPerson]) -> dict[int, float]:
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

            previous_pose = self.previous_pose.get(person.track_id)

            if previous_pose is None:

                self.previous_pose[person.track_id] = person.pose

                continue

            previous_left, previous_right = PoseAnalyzer.wrists(previous_pose)

            if (
                previous_left is None
                or previous_right is None
            ):

                self.previous_pose[person.track_id] = person.pose

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

                print(
                    f"ID {person.track_id} | "
                    f"Hand Speed: {hand_speed:.1f} px | "
                    f"Hand Risk: {hand_speed_scores[person.track_id]:.3f}"
                )

            self.previous_pose[person.track_id] = person.pose

        return hand_speed_scores
    

    def _compute_arm_extension_risk(self, people: list[TrackedPerson]) -> dict[int, float]:
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