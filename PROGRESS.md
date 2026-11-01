# PROGRESS

## Phase status
| Phase | Title | Status | Tag |
|---|---|---|---|
| 0 | Bootstrap | done | phase-0 |
| 1 | Documentation | done | phase-1 |
| 2 | Pose & geometry core | done | phase-2 |
| 3 | Reps, tempo, DTW, rules, offline analysis | done | phase-3 |
| 4 | Data tooling | done | phase-4 |
| 5 | Backend | next (waiting for dataset decision) | |
| 6 | Frontend | todo | |
| 7 | ML training & evaluation (MPS) | todo | |
| 8 | Reports, progress, gamification polish | todo | |
| 9 | Quality, CI, Docker, benchmarks | todo | |
| 10 | Final docs | todo | |
| 11 | Core ML / iOS (optional, only on request) | not requested | |

## Phase 0 tasks
- [x] git init (branch main), uv package `rehabpose` (src layout, uv_build, Python 3.12)
- [x] runtime deps, extras `[rtm]` `[ablation]` `[coreml]`, dev group, uv.lock
- [x] ruff / mypy (strict overrides for core, api) / pytest / coverage config in pyproject.toml
- [x] pre-commit (pre-commit-hooks + local uv-run ruff/mypy hooks)
- [x] .gitignore, Apache-2.0 LICENSE, README stub with disclaimer
- [x] settings.py (pydantic-settings, REHABPOSE_ env, configs/app.yaml), CLI stubs (8 subcommands)
- [x] CLAUDE.md, .claude/settings.json PostToolUse ruff-format hook
- [x] smoke tests, verification, commit + tag phase-0

## Phase 1 tasks
- [x] docs/01–09 from spec Sections 2, 3, 6 (overview, SRS with FR/NFR + user stories, architecture,
      exercise spec, data plan, ML/eval plan, testing, deployment, ethics/privacy)
- [x] Mermaid: component flowchart, live + upload sequences, ER diagram, rep FSM state diagram
- [x] ADR 0001 (landmarks over WebSocket), ADR 0002 (CPU-only torch lock)
- [x] stubs docs/10_model_card.md, docs/11_datasheet.md ("NOT YET MEASURED")
- [x] tests/unit/test_docs.py: links + anchors, Mermaid structure, table widths, disclaimer verbatim

## Phase 4 tasks
- [x] data/schema.py annotation v1 (pydantic, angle consistency check, JSON Schema)
- [x] data/labeling: pure LabelState + OpenCV tool (injectable UI) + `rehabpose label`
- [x] data/loaders: base (license gate, missing/layout errors, YAML layout override), own + 7 public
- [x] data/joint_maps.py Kinect v2 25 and Plug-in Gait -> MediaPipe UPPER13, virtual camera
- [x] data/splits.py held-out test + GroupKFold/LOSO, leakage checks, YAML plans
- [x] data/tasks.py stats + extract-all; `rehabpose data list|info|stats|extract-all|splits|schema`
- [x] scripts/download_{zenodo,mendeley,kaggle_trsp}.sh (NOT run; refuse without --yes)
- [ ] manual check of the labeling window on data/demo/label_demo.mp4 (owner)

## Phase 3 tasks
- [x] core/exercise.py pydantic YAML models; configs/exercises/*.yaml (5), synthetic min-jerk references
- [x] core/reps.py hysteresis FSM (arming, start re-anchoring, peak hysteresis, unknown reset)
- [x] core/tempo.py causal velocity; core/dtw.py Sakoe-Chiba DTW + similarity
- [x] core/rules.py calibration, frame/rep rules, persistence, unknown handling, labels
- [x] core/feedback.py priority, 0.5 s debounce, 1 cue / 3 s speech
- [x] core/engine.py SessionEngine (setup/view/calibration cues, events, summary)
- [x] render.py overlay, offline.py + `rehabpose analyze` (annotated.mp4, reps.csv, events.csv, summary.json)
- [x] synthetic rep-sequence generator; real-MediaPipe analyze test on a rendered mannequin set

## Phase 2 tasks
- [x] core/landmarks.py (Frame, LM indices, side selection, JSON with WS v1 field names)
- [x] pose/base.py PoseEstimator protocol; pose/mediapipe_pose.py (Tasks API, VIDEO, CPU default)
- [x] pose/model_files.py (official URLs, resolve, consent-only download, 50 MB cap)
- [x] core/filters.py OneEuroFilter (vectorized, gap reset, NaN pass-through), FrameSmoother, savgol helper
- [x] core/angles.py (2D/3D angle primitives, all measures, visibility-gated compute_measures)
- [x] data/video_io.py, data/extract.py + `rehabpose extract` CLI (parquet + provenance metadata)
- [x] tests/fixtures/synthetic.py (exact-angle skeletons, views, noise, occlusion, stick-figure video)
- [ ] real MediaPipe run: skipped until a model file is in data/models/ (needs owner OK)

## Decisions
- 2026-11-01: Node 24.19 (installed LTS) instead of 22. `.nvmrc` = 24, frontend `engines.node >= 22`.
- 2026-11-01: The uv lock is limited to darwin-arm64 + linux. Linux torch resolves from the
  download.pytorch.org CPU index, so there are no CUDA wheels. This is needed for Docker/CI in phase 9.
  ADR to be written in phase 1 (0002).
- 2026-11-01: `nvidia-ml-py` (pulled in only by ultralytics in `[ablation]`) is excluded via
  `tool.uv.override-dependencies`. `import ultralytics` was checked and works without it.
- 2026-11-01: mypy `strict` can't be set per module, so the strict flag set is listed explicitly for
  `rehabpose.core.*` / `rehabpose.api.*`.
- 2026-11-01: pre-commit ruff/mypy run as local `uv run` hooks so their versions match uv.lock.
- 2026-11-01 (P1): Mermaid is checked structurally by tests/unit/test_docs.py (diagram type, balanced
  quotes/brackets, declared sequence participants, subgraph/end pairs). mermaid-cli (Chromium) was not
  installed. Diagrams have not been rendered yet; check them on GitHub or with mermaid-cli if approved later.
- 2026-11-01 (P1): Proposed rep label precedence: compensated > too_fast > incomplete_rom > correct,
  mirroring cue priority (docs/04 §8). To be confirmed in phase 3.
- 2026-11-01 (P1): FSM thresholds for exercises other than shoulder_flexion reuse flexion's values
  (30/15/40) and are marked provisional (docs/04 §4). External-rotation target and forward-reach FSM
  units are still to be decided in phase 3.
- 2026-11-01 (P2): Normalization unit = `body_scale` = max(shoulder width, 0.7 × torso length).
  Pure shoulder width collapses to ~0 in sagittal 2D views, which made One Euro speeds explode:
  frame-smoother jitter reduction was ~45 % and failed the test. With the floor the test passes.
  Documented in docs/04 §2.
- 2026-11-01 (P2): One Euro step lag is defined as the time to reach 63 % of the step (one time
  constant). Measured in tests/unit/test_filters.py at 30 fps with default params: 100 / 67 / 33 ms
  for steps of 0.2 / 0.5 / 1.0 body-scale units. Time to 90 % for a 0.2 step is 233 ms (simulated
  2026-11-01), so small movements settle more slowly. Revisit params in phase 3 if rep timing suffers.
- 2026-11-01 (P2): Landmark parquet stores lm/world as variable-size list<float32> (132 values,
  validated on load). Parquet cannot round-trip nulls in fixed-size lists.
- 2026-11-01 (P2): MediaPipe 1.2.0 API matched the spec (inspected: BaseOptions.Delegate CPU/GPU,
  RunningMode.VIDEO, detect_for_video(image, timestamp_ms), result.pose_landmarks /
  pose_world_landmarks). No ADR needed. World landmarks copy the image-landmark visibility.
- 2026-11-01 (P2): Synthetic skeletons use hip width = shoulder width so that the S→H line is parallel
  to the trunk axis and the expected angles are exact by construction (documented in synthetic.py).
- 2026-11-01 (P3): Rep label precedence compensated > too_fast > incomplete_rom > correct (owner
  accepted the default).
- 2026-11-01 (P3): Rep FSM arming. A rep only starts after the primary has been seen ≤ start_below
  (at set start and after a > 0.5 s unknown reset), so half-observed reps are not counted. Risk: if a
  real elbow at rest has visibility < 0.6, the first rep is missed. This was seen on the cartoon
  fixture (elbow visibility ~0.06 at rest). Check on real recordings in phase 4/7.
- 2026-11-01 (P3): Experimental rules (PLANE_DRIFT) are off by default
  (EngineConfig.experimental_rules). Real MediaPipe world depth raised PLANE_DRIFT at every peak of
  correct reps on the synthetic mannequin video.
- 2026-11-01 (P3): View heuristic thresholds: shoulder/torso ratio ≥ 0.45 frontal, ≤ 0.25 sagittal.
  Rep-level flags stay displayed for 2.0 s. DTW resamples both curves to 64 samples (10 % band).
- 2026-11-01 (P3): Exercise schema lives in rehabpose.core.exercise (docs/04 §10 updated). Added
  measures plane_drift_m, elbow_away, and setting exercises_dir.
- 2026-11-01 (P3): The engine perf test is marked `no_cover`. Coverage tracing slows it ~2.4x
  (2.32 s vs 0.98 s measured).
- 2026-11-01 (P4): Public dataset file layouts are ASSUMED (no real files seen). Every loader has
  layout_verified=False, logs a warning, and can be re-pointed via configs/datasets/<name>.yaml.
  Native exercise codes are not mapped to RehabPose ids until verified. UI-PRMD Kinect joint order
  is not assumed (joint_format unknown_<J>). dib_upper has no subject ids under the assumed layout,
  so no subject-wise splits are possible.
- 2026-11-01 (P4): UPPER13 = MediaPipe {0, 7, 8, 11-16, 19, 20, 23, 24}. Kinect nose/ears <- Head.
  PiG hips <- mean(ASI, PSI). Approximations are listed in docs/05 §2.
- 2026-11-01 (P4): Labeling angle measures are limited to true 3-point angles (upper_arm_trunk_angle,
  elbow_interior). Rep flags in saved labels are derived from overlapping compensation segments.
- No TodoWrite tool is available in this environment. Per-phase task checklists live in this file.

## Pinned key versions (resolved 2026-11-01, see uv.lock)
mediapipe 1.2.0 (Tasks API `PoseLandmarker` present), torch 2.15.0, numpy 2.5.4, opencv-python 5.0.0.93,
fastapi 0.162.0, pydantic-settings 2.16.2, mlflow 3.17.2, lightgbm 4.7.0.

## Open issues
- Waiting on the owner: which datasets are downloaded/approved (phase 4 STOP question).
- Default TOO_FAST peak velocity (90 °/s) flags minimum-jerk reps faster than about 6 s for a 150°
  ROM. That is consistent with the 45 °/s "controlled" reference, but therapists may find it strict.
  It is editable per user (phase 5/8).
- Real MediaPipe on the rendered mannequin raised ELBOW_BEND (interior < 160°) near the overhead peak
  on all 3 reps, although the ground-truth arm is straight. This may be a cartoon artefact. Validate
  the threshold on real recordings before trusting it.
- No MediaPipe model downloaded yet. The real-model test is skipped. To enable:
  `uv run rehabpose extract <video> --model full --download-model`, or download the URL printed by
  the CLI into data/models/.
- lightgbm on macOS may need `libomp` from Homebrew at import time (not tested yet, phase 7). Ask
  before running `brew install libomp`.
- ffmpeg is not installed (needed only for the optional GIF script in phase 10). Ask before installing.

## Resume
```bash
cd /Users/admin/Projects/RehabPos
uv sync
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src
git log --oneline --decorate | head
```
