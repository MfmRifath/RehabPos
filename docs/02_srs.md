# 02 — Software requirements specification

Status: draft v1 (phase 1). Requirement IDs are stable. Tests and docs refer to them.
Priority uses MoSCoW: **M**ust, **S**hould, **C**ould.

## 1. Purpose and context
RehabPose measures upper-limb rehabilitation exercises from a single RGB camera and gives real-time
form feedback. Background is in [01_overview.md](01_overview.md), and design is in
[03_architecture.md](03_architecture.md). RehabPose is not a medical device (see
[09_ethics_privacy.md](09_ethics_privacy.md)).

## 2. Functional requirements

### 2.1 Pose and measurement
| ID | Requirement | Pri |
|---|---|---|
| FR-01 | Estimate 33 MediaPipe landmarks (image-normalized + world metres) per frame in the browser (live) and in Python (upload). | M |
| FR-02 | Smooth landmarks causally with a One Euro filter. Reset the filter on gaps > 0.5 s. | M |
| FR-03 | Compute upper-arm–trunk angle, elbow interior/flexion angle, trunk tilt and hike ratio (definitions in [04](04_exercise_spec.md#3-measures)). | M |
| FR-04 | Report a measurement as `unknown` when any required landmark has visibility < 0.6, and emit no corrective cue from it. | M |
| FR-05 | Calibrate per set from the first 1.0 s at rest: baseline trunk tilt, ear–shoulder distance, shoulder width. | M |
| FR-06 | Select the exercising side (L/R) and the expected camera view (sagittal/frontal) per exercise. | M |

### 2.2 Repetitions and form
| ID | Requirement | Pri |
|---|---|---|
| FR-10 | Count reps with a hysteresis state machine rest → ascending → peak → descending → rest, with thresholds from YAML. | M |
| FR-11 | Reject reps shorter than 0.8 s or with peak < start threshold + 20°. | M |
| FR-12 | Record per rep: idx, t_start, t_peak, t_end, peak angle, duration, peak angular velocity, flags, label, DTW similarity to a reference rep. | M |
| FR-13 | Evaluate the rules per exercise (ELBOW_BEND, TRUNK_LEAN, SHOULDER_HIKE, INCOMPLETE_ROM, TOO_FAST, SHOULDER_DRIFT, PLANE_DRIFT, ELBOW_AWAY, TRUNK_ROTATION), with severity and confidence. | M |
| FR-14 | Show at most one cue, chosen by priority: setup/visibility > TRUNK_* > SHOULDER_HIKE > ELBOW_* > TOO_FAST > INCOMPLETE_ROM. | M |
| FR-15 | Debounce cues (the condition must persist ≥ 0.5 s) and rate-limit spoken cues to 1 per 3 s. | M |
| FR-16 | Optionally fuse rule flags with a temporal ML model (`source: model`), using a configurable OR with a probability threshold. | S |

### 2.3 Live session (frontend)
| ID | Requirement | Pri |
|---|---|---|
| FR-20 | Show a first-launch disclaimer modal. The user must acknowledge it before any session. | M |
| FR-21 | Run a setup check: camera permission, person fully visible, view orientation heuristic, distance hint. | M |
| FR-22 | Overlay the skeleton, highlighted active joints, angle arc + value, target band, rep counter, phase and cue banner (coloured by severity). | M |
| FR-23 | Speak cues with `speechSynthesis`, show captions, and provide a mute toggle. Keyboard: space = start/pause, m = mute. | M |
| FR-24 | Provide a toggleable HUD with client FPS, inference ms and round-trip latency. | S |
| FR-25 | Gamification: points per quality rep, combo for consecutive clean reps, badges, streaks. | C |

### 2.4 Sessions, reports, progress
| ID | Requirement | Pri |
|---|---|---|
| FR-30 | Persist users (pseudonymous), targets, sessions, reps and events in SQLite. | M |
| FR-31 | Generate a session report as JSON and PDF: summary, ROM vs target chart, per-rep table, flags timeline, disclaimer, model/version info. | M |
| FR-32 | Show progress per user and exercise: ROM trend, reps, quality score, streaks, badges. | S |
| FR-33 | Let a therapist set target ROM, minimum rep duration and thresholds per user and exercise. | S |
| FR-34 | Upload a video, process it in the background, poll job status, and return the session plus an annotated MP4. | M |
| FR-35 | Export all of a user's data as JSON, and delete a user with cascade. | M |

### 2.5 Research tooling
| ID | Requirement | Pri |
|---|---|---|
| FR-40 | CLI `extract`, `analyze`, `data`, `train`, `evaluate`, `bench`, `label`, `serve`. | M |
| FR-41 | Dataset loaders behind a common interface. Each requires explicit licence acceptance and never auto-downloads. | M |
| FR-42 | Subject-wise splits (GroupKFold / LOSO + held-out test subjects) with a leakage test. | M |
| FR-43 | Experiment tracking in a local MLflow file store. Reports cite run ids. | M |
| FR-44 | An OpenCV labeling tool for rep boundaries, labels, compensation segments and 3-click angle annotation. | S |

## 3. Non-functional requirements
| ID | Category | Requirement | Verification |
|---|---|---|---|
| NFR-01 | Latency | Server processing per landmark message p95 ≤ 10 ms | WS replay benchmark (phase 5/9) |
| NFR-02 | Throughput | Engine processes 10k frames in < 2 s on CPU | perf unit test (phase 3) |
| NFR-03 | Frame rate | Live client ≥ 20 FPS on an Apple M3-class laptop | manual HUD reading by the owner (phase 6) |
| NFR-04 | ML latency | Temporal model ≤ 5 ms per window | evaluate.py latency measurement (phase 7) |
| NFR-05 | Privacy | No video leaves the browser in live mode. Only landmarks are transmitted. | architecture + WS schema review, integration test |
| NFR-06 | Privacy | User IDs are pseudonymous. Export and delete are available. | API tests (phase 8) |
| NFR-07 | Platform | Apple Silicon macOS. torch device mps → cpu. No CUDA code paths or packages. | lock-file scan, code review |
| NFR-08 | Portability | Single Docker image (CPU-only) and docker-compose. | `docker compose build` (phase 9) |
| NFR-09 | Offline | Works without internet after setup. Model/wasm files are served locally. | production build check (phase 6) |
| NFR-10 | Robustness | Visibility gating, filter reset on gaps, reconnecting WS client, backpressure (latest frame wins). | unit + integration tests |
| NFR-11 | Configurability | All thresholds live in YAML and are validated by pydantic. | config tests |
| NFR-12 | Quality | ruff + mypy (strict core/api) clean. Coverage ≥ 80 % for `rehabpose.core`. | CI (phase 9) |
| NFR-13 | Licensing | Core is Apache-2.0. AGPL code only in the `[ablation]` extra behind a lazy import. | import test, dependency review |
| NFR-14 | Honesty | No reported number without a reproducible command and output path. | review checklist in [07](07_testing_plan.md) |
| NFR-15 | Accessibility | Cues are both spoken and captioned. Colour is never the only signal. Keyboard operable. | manual check (phase 6) |

## 4. User stories
Acceptance criteria are written as *Given / When / Then* where useful.

### Patient
- **US-P1**: As a patient, I want to pick my exercise and side, so that the coach measures the right joint.
- **US-P2**: As a patient, I want a setup check that tells me to step back or turn sideways, so that
  measurements are reliable. *Given* my hips are out of frame, *when* I start, *then* I see a setup
  cue and no form cues.
- **US-P3**: As a patient, I want to hear one short cue when I lean or bend my elbow, so that I can
  correct myself without looking at the screen.
- **US-P4**: As a patient, I want to see my rep count and how close I got to my target, so that I stay motivated.
- **US-P5**: As a patient, I want streaks and badges, so that I keep doing my programme.
- **US-P6**: As a patient, I want to export or delete my data, so that I stay in control of it.

### Therapist
- **US-T1**: As a therapist, I want to set target ROM and thresholds per patient, so that feedback
  matches their stage of recovery.
- **US-T2**: As a therapist, I want a PDF report per session with per-rep ROM, tempo and compensations,
  so that I can review home practice.
- **US-T3**: As a therapist, I want ROM trends over weeks, so that I can judge progress.
- **US-T4**: As a therapist, I want to see that the tool is not a medical device and how reliable each
  measure is, so that I interpret results correctly.

### Demo viewer
- **US-D1**: As a demo viewer, I want the app to work in my browser within a minute, so that I can
  judge the project quickly.
- **US-D2**: As a demo viewer, I want an FPS/latency HUD and an architecture page, so that I can assess
  the engineering.
- **US-D3**: As a demo viewer, I want to upload a short clip and get an annotated video back, so that I
  can try it without a webcam.

### Researcher
- **US-R1**: As a researcher, I want loaders for public rehab datasets with their licences printed, so
  that I can reproduce experiments legally.
- **US-R2**: As a researcher, I want subject-wise cross-validation with MLflow run ids, so that results
  are honest and reproducible.
- **US-R3**: As a researcher, I want ablations (pose backend, 2D vs 3D, smoothing, window length), so
  that I can understand which design choices matter.
- **US-R4**: As a researcher, I want a model card and a datasheet, so that I know the limitations.

## 5. Constraints
- Python 3.12 (uv), Node 24 LTS, MediaPipe Tasks API only (no legacy `mp.solutions`).
- Local-first: SQLite, a local MLflow file store, no external services.
- Data handling rules: [05_data_plan.md](05_data_plan.md) and [09_ethics_privacy.md](09_ethics_privacy.md).
