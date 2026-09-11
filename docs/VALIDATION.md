# Build validation

## Automated checks

Executed in the offline build environment:

- `python -m unittest discover -v` — **12/12 passed**
- `python -m compileall -q .` — passed
- `git diff --check` — passed
- Headless three-frame annotated MP4 pipeline with injected tracker — passed
- JSONL alert serialization — passed
- CLI help/import without Ultralytics or MediaPipe installed — passed
- Renderer inspection using deterministic synthetic tracks — passed after removing overlay collisions

## Covered behavior

- Person-height normalization across different apparent scales
- Time-normalized approach, body, and wrist motion
- Arm-extension gating
- Close-interaction duration accumulation and reset
- Missing-pose proximity fallback
- Asymmetric score decay
- Sustained alert confirmation
- Hysteresis around thresholds
- Direct state transitions
- Valid event payloads including the measured score
- Webcam/video CLI parsing
- Headless annotated-video writing and cleanup

## Environment limitation

The sandbox had OpenCV and NumPy but did not have Ultralytics, MediaPipe, model weights, a webcam, or an approved test video. Therefore, full YOLO/ByteTrack/MediaPipe inference must receive a short hardware smoke test after installing `requirements.txt` on Python 3.10–3.12. No model-performance claim is made.

## Recommended acceptance test

```bash
python -m unittest discover -v
python main.py --source path/to/consented_test.mp4 --output outputs/smoke.mp4 --headless --max-frames 300
```

Inspect track continuity, skeleton placement, processing speed, event JSONL, and output duration before merging.
