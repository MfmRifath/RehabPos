# RehabPose — guide for Claude Code

RehabPose is a real-time, camera-only rehabilitation exercise coach, built as a portfolio-grade
Apache-2.0 open-source research prototype (not a medical device). The browser runs MediaPipe pose
estimation and streams only landmarks to a FastAPI backend. There, a core engine (One Euro smoothing
→ joint angles → rep state machine → YAML rules, plus an optional temporal model) returns form feedback,
counts reps and writes session reports. Uploaded videos are processed offline with Python MediaPipe.
The work proceeds in phases. PROGRESS.md is the live status. The original build prompt is the spec,
and `docs/SPEC_SOURCE.md` overrides it if that file exists.

## Key commands
```bash
uv sync                                          # install (add --extra rtm|ablation|coreml as needed)
uv run pytest -q                                 # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy src                                  # strict for rehabpose.core / rehabpose.api
uv run rehabpose --help                          # CLI: extract analyze serve data train evaluate bench label
uv run rehabpose serve                           # API on 127.0.0.1:8000 (phase 5+)
uv run pre-commit run --all-files
cd frontend && npm run dev | npm test | npm run build   # phase 6+ (Node 24, see .nvmrc)
```

## Non-negotiable rules (condensed)
- Apple Silicon only. Torch device = "mps" if available else "cpu" (`rehabpose.settings.torch_device`).
  No CUDA, nvidia-*, onnxruntime-gpu, tensorrt. Linux torch comes from the CPU index (pyproject `tool.uv`).
- No fabricated results. Every number in docs/README/reports comes from a command that was actually run,
  with its output path (plus the MLflow run id for ML). Otherwise write "NOT YET MEASURED".
- Never commit data, videos, weights > 5 MB or personal data. data/, mlruns/, *.mp4, *.mov, *.parquet,
  *.pt, *.onnx are git-ignored. The only exception is tiny synthetic fixtures (< 1 MB) in tests/fixtures/.
- ASK BEFORE: downloading any dataset/model > 50 MB, accepting a dataset license, brew installs, or
  anything that needs credentials. Download scripts are written, never auto-run. Small pose models
  (< 50 MB) may go to data/models/ after telling the owner.
- Ultralytics (AGPL) is allowed only in the `[ablation]` extra and in `pose/yolo_pose.py` via lazy import.
  Core code never imports it.
- The disclaimer (`rehabpose.DISCLAIMER`, verbatim) goes in the README, the UI first-launch modal and every report.
- Verify, don't assert: run each phase's verification commands and paste the key output in the report.
- Keep PROGRESS.md updated. At the end of each phase: a Conventional Commit and tag `phase-N`. No push
  unless asked. STOP after each phase and wait for "continue".
- If a library API differs from the spec, inspect the installed source, adapt, and record an ADR in docs/adr/.
- Prefer small, typed, tested modules. When scope is unclear, ask.

## Coding standards
- Type hints everywhere. numpy arrays use `numpy.typing.NDArray`. mypy is strict for core/ and api/.
- core/ is pure functions where possible. State lives only in engine/filters/FSM classes.
- Thresholds live in YAML (configs/), validated by pydantic models. No magic constants.
- Use the `logging` module (rich handler in the CLI). No print in library code.
- Every public function in core/ has a unit test. Tests use synthetic fixtures only (no network, camera or real data).
- Docstrings: a one-line summary plus units (degrees, seconds, normalized units).
- Line length 100. Ruff rules E,F,I,UP,B,SIM,N. A PostToolUse hook ruff-formats edited .py files.

## Architecture (one line)
Browser webcam → MediaPipe Tasks Vision (GPU) → landmarks over WS `/ws/session/{sid}` (protocol v1)
→ SessionEngine (filters → angles → reps → rules/model → feedback) → cue JSON → UI banner + TTS.
Upload mode: POST video → background job → Python MediaPipe (VIDEO mode, CPU) → same engine
→ SQLite (SQLModel) + annotated MP4 + JSON/PDF report. MLflow uses the local file store.

## Layout
`src/rehabpose/{core,pose,data,ml,api,bench}`, `configs/{app.yaml,exercises,train}`, `docs/` (+ `adr/`),
`frontend/` (Vite React TS), `tests/{unit,integration,fixtures}`, `scripts/` (never auto-run), `reports/`.

## When compacting
Preserve the PROGRESS.md status, the list of modified files and the test commands.
