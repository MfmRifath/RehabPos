# ADR 0002 — Platform-limited uv lock with CPU-only torch on Linux

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** project owner, Claude Code

## Context
The project targets Apple Silicon (MPS/CPU) and must never add CUDA, `nvidia-*`, onnxruntime-gpu or
tensorrt. `uv.lock` is universal: it resolves dependencies for every platform at once. On Linux, the
default PyPI `torch` wheel depends on many `nvidia-*` CUDA packages, so a naive lock contains them
even when developing on a Mac. Linux resolution is still needed for the Docker image and the
`ubuntu-latest` CI job (phase 9). Separately, `ultralytics` (only in the optional `[ablation]` extra)
declares `nvidia-ml-py`, a pure-Python NVML telemetry binding.

## Decision
In `pyproject.toml`:
- `tool.uv.environments` limits resolution to `darwin/arm64` and `linux`.
- An explicit index `pytorch-cpu` (`https://download.pytorch.org/whl/cpu`) is used for `torch` only
  when `sys_platform == 'linux'`. macOS uses the PyPI wheel, which includes MPS support.
- `tool.uv.override-dependencies = ["nvidia-ml-py; sys_platform == 'never'"]` removes the NVML
  binding from the lock.

## Verification (run 2026-11-01, phase 0)
- `grep -ciE 'nvidia|cuda' uv.lock` → 1. The only match is the override declaration itself.
- The lock contains `torch 2.15.0` from PyPI (macOS) and `torch 2.15.0+cpu` from the CPU index (Linux).
- With the override, `uv run --extra ablation python -c "import ultralytics"` imported version 8.4.197
  successfully.
- `torch.backends.mps.is_available()` → `True` on the owner's Apple M3 Pro.

## Consequences
- No CUDA/NVIDIA packages can be installed from this lock, on any platform.
- Windows and Intel macOS are unsupported. `uv sync` there fails to find a matching environment,
  which is intended.
- Ultralytics GPU telemetry features that call NVML would fail. They are irrelevant because there is
  no NVIDIA GPU, and ultralytics imports NVML lazily.
- Upgrading torch must keep the CPU index for Linux. Re-run the grep check after every `uv lock`.
