# ADR 0004 — ML tooling: MLflow on SQLite, ONNX preferred online, single-threaded LightGBM

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** project owner, Claude Code

## Context
Three library behaviours found in phase 7 differ from what the project spec assumed. Versions:
MLflow 3.17.2, torch 2.15.0, LightGBM 4.7.0 on an Apple M3 Pro.

1. **MLflow refuses the file store.** `file:./mlruns` raises `MlflowException` ("in maintenance
   mode") unless `MLFLOW_ALLOW_FILE_STORE=true` is set. The spec asked for a "local file store".
2. **TorchScript is deprecated.** `torch.jit.trace` and `torch.jit.load` emit `FutureWarning`s
   that point to `torch.export`. The spec asks for TorchScript and ONNX exports.
3. **torch + multi-threaded LightGBM segfaults.** LightGBM's wheel links Homebrew's `libomp`
   (`/opt/homebrew/opt/libomp/lib`) and torch bundles its own `libomp.dylib`. With torch imported
   first, `LGBMClassifier.fit` exits with code 139. With `n_jobs=1` it does not crash.

## Decision
1. Track experiments in a local SQLite database, `mlruns/mlflow.db`, with artifacts as files under
   `mlruns/artifacts/`. This keeps the intent of the spec: local, no server, git-ignored.
2. Keep both exports as the spec asks. `TemporalFlagModel` uses **ONNX Runtime when available** and
   falls back to TorchScript. Export parity is checked for both on every run.
3. `make_lgbm` pins `n_jobs=1`. The per-rep feature table is small, so this costs little.
   `test_torch_then_lightgbm_does_not_segfault` runs the failing order in a subprocess, so a
   regression fails the test instead of killing pytest.

## Consequences
- `onnx`, `onnxscript` and `onnxruntime` (CPU build) are dev dependencies. A production install
  without them uses TorchScript until it is removed from torch; at that point onnxruntime must
  become a runtime dependency of the online model.
- Machines without Homebrew's `libomp` have not been tested with LightGBM. The earlier open question
  about `brew install libomp` is resolved for this machine only (it was already installed).
- `n_jobs=1` is a workaround, not a fix. Any other OpenMP library loaded with torch may need the
  same care.
