# AI Escalation Predictor

A real-time AI-powered behavioral risk assessment system that analyzes human interactions using computer vision and estimates the likelihood of conflict escalation through pose estimation, motion analysis, and behavioral analysis.

---

## Overview

AI Escalation Predictor is a computer vision project designed to assess the risk of physical conflict by analyzing human behavior in real time.

Unlike traditional fight detection systems that react after violence begins, this project focuses on identifying behavioral patterns that may indicate an interaction is escalating. The system combines person detection, pose estimation, motion analysis, interpersonal distance measurement, and behavioral modeling to generate a dynamic Conflict Risk Score.

---

## Objectives

- Detect and track multiple people in real time.
- Analyze body posture and movement patterns.
- Monitor interpersonal distance and interaction duration.
- Estimate the probability of conflict escalation.
- Generate explainable conflict risk scores.
- Trigger alerts for sustained high-risk behavior.

---

## System Architecture

```text
                       Webcam / CCTV Feed
                               │
                               ▼
                  ┌─────────────────────────┐
                  │     Person Detection    │
                  │      (YOLOv8 Model)     │
                  └─────────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │     Person Tracking     │
                  │   Persistent IDs        │
                  └─────────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    Pose Estimation      │
                  │      (MediaPipe)        │
                  └─────────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   Behavior Analysis     │
                  │ • Motion                │
                  │ • Distance              │
                  │ • Orientation           │
                  │ • Gestures              │
                  └─────────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │      Risk Engine        │
                  │ Conflict Probability    │
                  └─────────────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Dashboard & Alerts      │
                  │ Recording │ Alarm       │
                  └─────────────────────────┘
```

---

## Planned Features

- Real-time webcam processing
- Multi-person detection
- Human pose estimation
- Persistent person tracking
- Motion analysis
- Interpersonal distance analysis
- Aggressive gesture detection
- Conflict risk scoring
- Alert generation
- Evidence recording
- Interactive monitoring dashboard

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 |
| Pose Estimation | MediaPipe |
| Numerical Computing | NumPy |
| GUI | CustomTkinter |
| Deep Learning (Planned) | PyTorch |

---

## Project Structure

```text
AI-Escalation-Predictor/
│
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .editorconfig
├── .gitignore
│
├── config/
├── detection/
├── tracking/
├── analysis/
├── alerts/
├── gui/
├── utils/
├── assets/
├── data/
├── docs/
├── logs/
└── tests/
```

---

## Development Roadmap

| Phase | Status |
|------|--------|
| Phase 1 – Project Setup | Completed |
| Phase 2 – Webcam Integration | Planned |
| Phase 3 – Person Detection | Planned |
| Phase 4 – Pose Estimation | Planned |
| Phase 5 – Person Tracking | Planned |
| Phase 6 – Behavior Analysis | Planned |
| Phase 7 – Conflict Risk Engine | Planned |
| Phase 8 – Dashboard | Planned |
| Phase 9 – Alert System | Planned |
| Phase 10 – Machine Learning-Based Risk Prediction | Planned |

---

## Current Status

**Version:** 1.0

The project foundation has been established. Development will continue with webcam integration, followed by person detection, pose estimation, behavioral analysis, and conflict risk estimation.

---

## Author

**August Kumar Sasmal**

B.Tech Computer Science & Engineering<br>
Manipal Institute of Technology, Manipal