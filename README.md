# AI Escalation Predictor

An explainable computer-vision **research prototype** that tracks people in webcam or recorded video, extracts observable interaction cues, and produces a temporally smoothed cue score with sustained alerts.

> **Important:** the score is not a probability of violence, a diagnosis, or evidence of intent. It has not been validated for safety, policing, employment, education, access control, or other high-impact decisions. A human must review the original footage and context.

## What works

- Webcam and video-file input
- YOLOv8 person detection with ByteTrack IDs
- Per-person MediaPipe pose estimation with short-lived pose caching
- Person-size and time-normalized feature extraction
- Explainable cues: proximity, approach speed, body movement, hand speed, arm extension, and close-interaction duration
- Multi-cue fusion and asymmetric temporal smoothing
- Alert hysteresis and sustained confirmation for Watch, Warning, and Critical states
- On-frame tracks, skeletons, cue scores, primary factors, FPS, and system status
- Optional annotated MP4 export
- Confirmed alert logging as JSON Lines
- Detection-only fallback with `--disable-pose`
- Dependency-free unit tests for scoring and alert logic plus a headless video-pipeline integration test

## Quick start

Python **3.10–3.12** is recommended because MediaPipe support can lag behind new Python releases.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

MediaPipe is pinned to **0.10.21** because version 0.10.31 removed the legacy `mp.solutions` API used by the pose adapter. NumPy is pinned to **1.26.4**, and both OpenCV packages are pinned to **4.11.0.86**; OpenCV 4.14 requires NumPy 2 and conflicts with this MediaPipe stack. Do not upgrade MediaPipe without migrating the adapter to MediaPipe Tasks.

The default YOLO model weights (`yolov8n.pt`) are downloaded by Ultralytics on first use. To avoid a network download, pass a local model path with `--model`.

### Webcam

```bash
python main.py --source 0
```

Press **Q** to exit.

### Recorded video

```bash
python main.py --source path/to/input.mp4 --output outputs/annotated.mp4
```

### Headless evaluation

```bash
python main.py \
  --source path/to/input.mp4 \
  --output outputs/annotated.mp4 \
  --events-log outputs/events.jsonl \
  --headless
```

### Useful options

```text
--source          Webcam index or video path
--output          Annotated MP4 destination
--events-log      Confirmed-event JSONL destination
--model           YOLO model name or local weight path
--device          cpu, mps, or CUDA device such as 0
--disable-pose    Detection/tracking/proximity fallback
--max-frames      Bounded smoke test or evaluation run
--headless        Disable the OpenCV window
--debug           Detailed logs
```

Run `python main.py --help` for the complete list.

## How scoring works

Each active track receives a bounded score from observable cues. Distances are divided by average person height, and motion is measured per second, reducing sensitivity to resolution and frame rate. Pairwise features use the strongest active interaction instead of simply adding risk for every nearby person. Multiple simultaneous cues can add a small synergy bonus. Scores rise faster than they decay to reduce flicker while preserving recovery.

The overlay calls this a **cue score**, not a probability. See [`docs/RISK_MODEL.md`](docs/RISK_MODEL.md) for formulas, defaults, and assumptions.

## Alerts

| State | Default entry score | Confirmation |
|---|---:|---:|
| Normal | below 0.25 | immediate |
| Watch | 0.25 | 1.50 s |
| Warning | 0.50 | 1.25 s |
| Critical | 0.75 | 0.75 s |

A 0.07 hysteresis margin prevents rapid state changes around thresholds. Confirmed non-normal states are appended to the JSONL event log with the track ID, score, severity, timestamp, and dominant cue.

## Tests

The core suite does not load YOLO weights, require a camera, or require MediaPipe:

```bash
python -m unittest discover -v
```

For development tooling:

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The included tests cover normalization, cue activation, smoothing, close-interaction accumulation, missing-pose fallback, alert confirmation, hysteresis, event serialization, CLI parsing, and headless annotated-video output.

## Evaluation before claiming performance

No accuracy, recall, or false-alarm claim is made. A meaningful evaluation requires consented, representative, labelled clips and a documented annotation protocol. Split recordings by scene or participant—not by frame—to avoid leakage. Report precision/recall, false alarms per hour, missed-event rate, alert latency, subgroup/scene breakdowns, and calibration limitations. See [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Privacy and responsible use

- Process only footage you have permission to use.
- Prefer local processing and short retention.
- Do not use identity recognition; this project only uses temporary track IDs.
- Do not infer intent, protected traits, mental state, or criminality.
- Never trigger force, punishment, denial of service, or another high-impact action automatically.
- Treat alerts as prompts for a trained human to inspect context.
- Document camera placement, lighting, occlusion, retention, and access controls.

## Project structure

```text
alerts/      confirmed state machine and JSONL events
analysis/    pose helpers and explainable temporal risk engine
app/         end-to-end application controller
camera/      webcam/video capture abstraction
config/      model defaults and runtime options
graphics/    OpenCV research overlay
models/      typed data models
pose/        MediaPipe adapter and landmark conversion
tracking/    YOLOv8 + ByteTrack adapter
tests/       deterministic unit and integration tests
docs/        scoring and evaluation documentation
```

## Known limitations

- Heuristic weights are engineering defaults, not learned or clinically validated parameters.
- A single RGB camera cannot reliably determine intent, depth, contact, speech, or off-camera context.
- Bounding-box height is only an approximate scale reference.
- Occlusion, unusual camera angles, crowds, low light, mobility aids, clothing, and tracking-ID switches can change scores.
- MediaPipe pose is run on cropped detections and may fail on partial bodies.
- Model downloads and full webcam inference were not available in the offline build environment; the adapters require local hardware validation after dependency installation.
- This version does not perform face recognition, identify individuals, save evidence clips, send remote notifications, or train a classifier.

## Research roadmap

1. Run the documented hardware smoke test on Python 3.11.
2. Collect a small, consented dataset with ordinary interactions and staged escalation cues.
3. Freeze an annotation protocol before tuning thresholds.
4. Record raw feature traces and conduct ablation studies.
5. Tune thresholds only on training/validation scenes; reserve held-out scenes for final reporting.
6. Add optional evidence clips only after defining retention and consent controls.
7. Consider a learned temporal model only when enough labelled data exists; retain this explainable baseline for comparison.

## Author

**August Kumar Sasmal**<br>
B.Tech Computer Science & Engineering<br>
Manipal Institute of Technology, Manipal
