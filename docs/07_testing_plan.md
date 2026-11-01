# 07 — Testing plan

## 1. Principles
- Tests sit next to behaviour. Every public function in `rehabpose.core` has a unit test.
- Tests are deterministic and offline: synthetic fixtures only, with no network, camera or real data in CI.
- Tests use exact ground truth: synthetic skeletons are generated with known joint angles, rep
  counts and compensations (`tests/fixtures/synthetic.py`, phase 2).
- Performance claims are tested and printed. Numbers in docs come from these runs (NFR-14).

## 2. Test levels
| Level | Location | Tooling | Examples |
|---|---|---|---|
| Unit | `tests/unit/` | pytest, hypothesis | angles, filters, FSM, DTW, rules, feedback, settings, schemas |
| Property | `tests/unit/` | hypothesis | angle invariance to translation/scale, DTW symmetry and identity, filter stability |
| Integration | `tests/integration/` | pytest, httpx, Starlette TestClient | WS replay of landmark JSONL, REST flows, upload job with a mocked estimator, PDF report |
| Performance | `tests/unit/test_perf_*.py` (marker `perf`) | pytest + `time.perf_counter` | engine 10k frames < 2 s, server p95 ≤ 10 ms, model ≤ 5 ms/window |
| ML sanity | `tests/unit/test_ml_*.py` | pytest | overfit-one-batch, subject-leakage, export parity |
| Frontend unit | `frontend/src/**/*.test.ts` | vitest | protocol types/guards, cue logic, reducers, rate limiter |
| Frontend E2E | `frontend/e2e/` | Playwright (smoke) | app loads, disclaimer modal, live page with mocked WS |
| Docs | `tests/unit/test_docs.py` | pytest | relative links and anchors resolve, Mermaid block structure, table column counts, disclaimer verbatim |

## 3. Acceptance checks per phase
| Phase | Check |
|---|---|
| 2 | angles within 0.5° on synthetic skeletons (2D and 3D). Translation/scale invariance. One Euro reduces noise std by ≥ 50 % and lags a step by < 150 ms at 30 fps. `extract` runs on a synthetic video (or a mocked estimator, with the reason stated). |
| 3 | rep-count MAE = 0 on synthetic sequences (5–15 reps, noise, pauses, partial reps). Positive and negative test per rule. Engine 10k frames < 2 s. |
| 4 | each loader passes on a tiny synthetic fixture in its format. Split test proves no subject overlap. Labeling tool launches on a fixture video (manual). |
| 5 | WS replay asserts rep count and flags. Server p95 ≤ 10 ms (printed). REST tests including a non-empty PDF. |
| 6 | `npm run build`. vitest passes. Playwright smoke with mocked WS. Owner's manual FPS reading. |
| 7 | overfit-one-batch, leakage test, a full CV run (real or clearly labelled synthetic), inference ≤ 5 ms/window. |
| 8 | aggregation, badges/streaks, delete-user cascade. Sample report in `reports/samples/`. |
| 9 | CI commands pass locally. `docker compose build` succeeds. Benchmarks are real or "NOT YET MEASURED". |
| 10 | markdown/link checks pass. No unmeasured numbers. |

## 4. Synthetic fixtures
- **Skeleton generator:** builds a 33-landmark pose from joint angles (shoulder elevation in a
  chosen plane, elbow flexion, trunk tilt, shoulder hike, trunk rotation). It also sets camera
  view, image size, additive noise, visibility dropout/occlusion, and dropped frames.
- **Sequence generator:** produces rep sequences from minimum-jerk angle profiles, with a
  configurable rep count, tempo, pauses, partial reps and scripted compensations. It returns exact
  ground truth for reps and flags.
- **Video fixture:** an OpenCV-drawn stick figure, small and generated at test time (not committed),
  used to exercise the extract pipeline.
- **Dataset fixtures:** a few frames per public format (Kinect 25-joint CSV/TXT, mocap, MediaPipe
  parquet), generated in tests or committed if < 1 MB.

## 5. Coverage and static analysis
- `pytest --cov=rehabpose --cov-fail-under=80` applies to `rehabpose.core` (enforced in CI, phase 9).
- `ruff check`, `ruff format --check` and `mypy src` (strict for core/api) are clean on every commit
  (pre-commit + CI).

## 6. Manual test scripts
Phases 4 (labeling tool) and 6 (live UI) include step-by-step manual checks with expected
observations. The owner runs them and reports the results. Claude never claims manual results.

## 7. Review checklist for numbers (NFR-14)
Before any number is added to README or docs:
1. Is there a command that produced it, recorded next to the number?
2. Is there an output file under `reports/` (and an MLflow run id for ML)?
3. Is the hardware/software context stated (chip, macOS, library versions)?
4. If any answer is no, write **NOT YET MEASURED**.
