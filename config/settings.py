"""
Application-wide configuration settings.
"""

# Camera
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720

# Object detection
PERSON_CLASS_ID: int = 0

# Pose estimation intervals
POSE_INTERVAL_SINGLE_PERSON: int = 2
POSE_INTERVAL_FEW_PEOPLE: int = 3
POSE_INTERVAL_MANY_PEOPLE: int = 4

# Minimum size required before running pose estimation
MIN_PERSON_WIDTH: int = 120
MIN_PERSON_HEIGHT: int = 200

# Tracking
TRACK_PERSIST: bool = True