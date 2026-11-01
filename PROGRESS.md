# PROGRESS

## Phase status
| Phase | Title | Status | Tag |
|---|---|---|---|
| 0 | Bootstrap | done | phase-0 |
| 1 | Documentation | next | |
| 2 | Pose & geometry core | todo | |
| 3 | Reps, tempo, DTW, rules, offline analysis | todo | |
| 4 | Data tooling | todo | |
| 5 | Backend | todo | |
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
- No TodoWrite tool is available in this environment. Per-phase task checklists live in this file.

## Pinned key versions (resolved 2026-11-01, see uv.lock)
mediapipe 1.2.0 (Tasks API `PoseLandmarker` present), torch 2.15.0, numpy 2.5.4, opencv-python 5.0.0.93,
fastapi 0.162.0, pydantic-settings 2.16.2, mlflow 3.17.2, lightgbm 4.7.0.

## Open issues
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
