# Explainable cue-score model

## Purpose

The engine ranks **observable interaction cues** for research review. It does not infer intent and its output is not a calibrated probability.

## Coordinate normalization

For each tracked person, the bounding-box height is used as a rough scale. Pair distance is:

```text
distance_ratio = pixel_distance(body_centers) / average_person_height
```

Motion features are measured in person-heights per second:

```text
speed = pixel_displacement / person_height / elapsed_seconds
```

This is more stable across resolution and frame rate than raw pixels per frame, but it does not replace camera calibration or true depth.

## Feature contributions

Each ramp is clipped to 0–1 and multiplied by a maximum contribution.

| Cue | Default activation | Full contribution | Maximum |
|---|---:|---:|---:|
| Proximity | below 1.80 heights | 0.55 heights | 0.18 |
| Approach speed | 0.15 heights/s | 1.00 heights/s | 0.16 |
| Body movement | 0.15 heights/s | 1.20 heights/s | 0.14 |
| Hand speed | 0.40 heights/s | 2.40 heights/s | 0.14 |
| Arm extension | 0.80 ratio | 0.98 ratio | 0.10 |
| Close duration | after 1.0 s | 5.0 s | 0.12 |
| Multi-cue bonus | 3+ active cues | bounded | 0.10 |

Arm extension is gated by hand movement so a static extended arm is not scored by itself. Pairwise features take the strongest interaction for each person, which limits crowd-size inflation.

## Temporal behavior

Raw contributions are summed and clipped to 1. The score uses exponential smoothing:

```text
alpha = 1 - exp(-dt / tau)
smoothed = previous + alpha * (raw - previous)
```

The rise time constant is 0.35 seconds and the fall time constant is 1.80 seconds. This allows prompt increases while reducing single-frame drops. Per-frame time deltas are capped at 0.50 seconds.

## Alert states

Watch, Warning, and Critical states require sustained scores. A 0.07 hysteresis margin is applied when scores decrease. The manager emits one event after each newly entered level is confirmed.

## Interpretation

A high result means several configured visual cues were active. It does **not** establish aggression, violence, intent, identity, blame, or what occurred outside the camera view. The feature breakdown should always be retained with evaluation results so reviewers can understand why a score changed.

## Tuning protocol

Do not tune constants on a final test set. First define labels and acceptable operating points. Split data by participant or scene. Tune only on training/validation scenes, freeze settings, and then report held-out results with uncertainty and failure cases.
