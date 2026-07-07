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
    RISK_PROXIMITY_SCORE
)


class RiskEngine:
    """
    Computes an escalation score.

    Current implementation is a placeholder that always
    returns zero until behavioral features are added.
    """

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

        for i, first in enumerate(people):

            for second in people[i + 1:]:

                distance = self._distance_between_people(
                    first,
                    second
                )

                if distance < RISK_DISTANCE_THRESHOLD - RISK_DISTANCE_BUFFER:

                    risk_scores[first.track_id] += RISK_PROXIMITY_SCORE
                    risk_scores[second.track_id] += RISK_PROXIMITY_SCORE

        return risk_scores