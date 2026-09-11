# Evaluation protocol

## Research question

Can the configured visual cues help a human reviewer find segments containing staged escalation-like movement without producing an unacceptable number of alerts during ordinary interaction?

This is narrower than predicting violence or intent.

## Data collection

Use only consented recordings. Include ordinary conversation, walking past others, handshakes, exercise, dancing, sports-like movement, occlusion, different body sizes, mobility aids, varied lighting, and staged escalation cues. Record camera height, frame rate, resolution, scene, and consent/retention details.

Do not collect real violence for this prototype. Do not scrape surveillance footage or sensitive recordings.

## Annotation

Define labels before tuning. At minimum:

- `ordinary`: no staged escalation cue
- `ambiguous`: context insufficient or annotators disagree
- `escalation_cue`: predefined staged visual pattern is present
- cue start/end timestamps
- visibility/occlusion quality

Use at least two annotators and report agreement. Exclude ambiguous clips from a primary binary metric but report them separately.

## Splitting

Split by scene and participant. Frames from one recording must never appear across training, validation, and test sets. Keep the test set untouched until thresholds are frozen.

## Metrics

Report:

- Precision and recall at each alert level
- False alerts per hour
- Missed-event rate
- Median and 90th-percentile alert latency
- Time spent in each alert state
- Results by scene, lighting, occlusion, camera angle, and participant grouping
- Confidence intervals where the sample size permits

Frame accuracy alone is misleading because neighboring frames are highly correlated.

## Baselines and ablations

Compare:

1. Proximity only
2. Motion only
3. All cues without temporal smoothing
4. Full explainable engine

Remove each cue in turn to identify which features improve or harm held-out results.

## Calibration and reporting

The cue score is not a probability unless it is calibrated against held-out labelled data. Do not label `0.8` as “80% chance of violence.” Publish the exact configuration, software versions, hardware, model weights, and test-scene selection criteria with any result.

## Hardware smoke test

After installing dependencies on Python 3.11:

```bash
python main.py --source path/to/consented_test.mp4 --output outputs/smoke.mp4 --headless --max-frames 300
python -m unittest discover -v
```

Verify that tracks persist, skeletons stay inside boxes, output duration matches input, event logs are valid JSONL, Q exits the webcam mode, and memory does not grow continuously after people leave the scene.
