"""CLI entry point for the AI Escalation Predictor research prototype."""

from __future__ import annotations

import argparse
import logging

from config.runtime import RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a webcam or video for explainable interaction-risk cues. "
            "Scores are research heuristics, not predictions of intent."
        )
    )
    parser.add_argument("--source", default="0", help="Webcam index or video-file path")
    parser.add_argument("--output", help="Optional annotated .mp4 output")
    parser.add_argument("--events-log", default="logs/events.jsonl", help="Confirmed alert JSONL")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path/name")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, 0, or mps")
    parser.add_argument("--headless", action="store_true", help="Do not open a display window")
    parser.add_argument("--disable-pose", action="store_true", help="Run detection-only fallback")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam width")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam height")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be greater than zero")
    config = RuntimeConfig.from_values(
        source=args.source,
        output=args.output,
        events_log=args.events_log,
        model=args.model,
        device=args.device,
        headless=args.headless,
        disable_pose=args.disable_pose,
        max_frames=args.max_frames,
        width=args.width,
        height=args.height,
    )
    from app.application import Application
    return Application(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
