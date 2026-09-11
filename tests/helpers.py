from __future__ import annotations

from models.pose_landmark import PoseLandmark
from models.pose_result import PoseResult
from models.tracked_person import TrackedPerson


def person(
    track_id: int,
    center_x: int,
    center_y: int = 260,
    height: int = 200,
    wrist_offset: int = 0,
    extended: bool = False,
    with_pose: bool = True,
) -> TrackedPerson:
    width = 90
    bbox = (
        center_x - width // 2,
        center_y - height // 2,
        center_x + width // 2,
        center_y + height // 2,
    )
    pose = None
    if with_pose:
        points = [PoseLandmark(center_x, center_y, 0.95) for _ in range(33)]
        points[11] = PoseLandmark(center_x - 25, center_y - 45, 0.95)
        points[12] = PoseLandmark(center_x + 25, center_y - 45, 0.95)
        points[23] = PoseLandmark(center_x - 20, center_y + 35, 0.95)
        points[24] = PoseLandmark(center_x + 20, center_y + 35, 0.95)
        if extended:
            points[13] = PoseLandmark(center_x - 55, center_y - 45, 0.95)
            points[15] = PoseLandmark(center_x - 90 + wrist_offset, center_y - 45, 0.95)
            points[14] = PoseLandmark(center_x + 55, center_y - 45, 0.95)
            points[16] = PoseLandmark(center_x + 90 + wrist_offset, center_y - 45, 0.95)
        else:
            points[13] = PoseLandmark(center_x - 35, center_y - 15, 0.95)
            points[15] = PoseLandmark(center_x - 20 + wrist_offset, center_y + 15, 0.95)
            points[14] = PoseLandmark(center_x + 35, center_y - 15, 0.95)
            points[16] = PoseLandmark(center_x + 20 + wrist_offset, center_y + 15, 0.95)
        pose = PoseResult(points)
    return TrackedPerson(track_id, bbox, 0.95, pose)
