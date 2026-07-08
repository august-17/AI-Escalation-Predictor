"""
Computes an escalation risk score for tracked people.
"""

from __future__ import annotations

import math

from analysis.pose_analyzer import PoseAnalyzer
from models.tracked_person import TrackedPerson
from config.settings import (
    RISK_DISTANCE_THRESHOLD,
    RISK_DISTANCE_BUFFER,
    RISK_PROXIMITY_SCORE,
    BODY_MOVEMENT_THRESHOLD,
    BODY_MOVEMENT_SCORE
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

        self._previous_body_centers: dict[int, tuple[int, int]] = {}


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
        people: list[TrackedPerson],
        risk_scores: dict[int, float]
    ) -> None:
        """
        Update risk scores based on the proximity of tracked people.
        """

        for i, first in enumerate(people):

            for second in people[i + 1:]:

                distance = self._distance_between_people(
                    first,
                    second
                )

                if distance < (
                    RISK_DISTANCE_THRESHOLD
                    - RISK_DISTANCE_BUFFER
                ):

                    risk_scores[first.track_id] += (
                        RISK_PROXIMITY_SCORE
                    )

                    risk_scores[second.track_id] += (
                        RISK_PROXIMITY_SCORE
                    )


    def _compute_body_movement_risk(
        self,
        people: list[TrackedPerson],
        risk_scores: dict[int, float]
    ) -> None:
        """
        Update risk scores based on body movement.
        """

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

                if movement > BODY_MOVEMENT_THRESHOLD:

                    risk_scores[person.track_id] += (
                        BODY_MOVEMENT_SCORE
                    )

            self._previous_body_centers[
                person.track_id
            ] = current

        self._previous_body_centers = {
            track_id: center
            for track_id, center
            in self._previous_body_centers.items()
            if track_id in active_ids
        }


    def compute(self, people: list[TrackedPerson]) -> dict[int, float]:
        """
        Compute a risk score for each tracked person.

        Args:
            people: List of tracked people.

        Returns:
            Dictionary mapping track IDs to risk scores.
        """

        risk_scores = {
            person.track_id: 0.0
            for person in people
        }

        self._compute_proximity_risk(people, risk_scores)

        self._compute_body_movement_risk(people, risk_scores)

        return risk_scores