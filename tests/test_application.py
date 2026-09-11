from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.application import Application
from config.runtime import RuntimeConfig
from main import build_parser
from tests.helpers import person


class FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def track(self, frame: np.ndarray):
        self.calls += 1
        offset = min(self.calls * 4, 20)
        return [
            person(1, 180 + offset, with_pose=False),
            person(2, 460 - offset, with_pose=False),
        ]


class FakeCamera:
    is_file = True

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.released = False

    def open(self) -> bool:
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0).copy()

    def get_fps(self) -> float:
        return 10.0

    def get_resolution(self):
        return 640, 360

    def release(self) -> None:
        self.released = True


class ApplicationTests(unittest.TestCase):
    def test_cli_accepts_webcam_or_video_options(self) -> None:
        args = build_parser().parse_args(
            ["--source", "clip.mp4", "--headless", "--disable-pose", "--max-frames", "12"]
        )
        self.assertEqual(args.source, "clip.mp4")
        self.assertTrue(args.headless)
        self.assertEqual(args.max_frames, 12)

    def test_headless_video_pipeline_writes_annotated_output(self) -> None:
        frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(3)]
        camera = FakeCamera(frames)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotated.mp4"
            config = RuntimeConfig(
                source="input.mp4",
                output=output,
                events_log=Path(directory) / "events.jsonl",
                headless=True,
                disable_pose=True,
                max_frames=3,
            )
            application = Application(config, camera=camera, tracker=FakeTracker())
            result = application.run()
            self.assertEqual(result, 0)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            capture = cv2.VideoCapture(str(output))
            success, rendered = capture.read()
            capture.release()
            self.assertTrue(success)
            self.assertGreater(int(rendered.sum()), 0)
        self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
