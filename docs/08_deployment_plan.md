# 08 — Deployment plan

RehabPose is local-first: it runs on the user's own machine. There is no cloud deployment in v1.

## 1. Native (development and demo)
Requires an Apple Silicon Mac, uv, Python 3.12 and Node 24 (`.nvmrc`).

```bash
uv sync                               # Python deps from uv.lock
uv run rehabpose serve                # FastAPI on http://127.0.0.1:8000
cd frontend && npm ci && npm run dev  # Vite on http://localhost:5173 (proxied / CORS to :8000)
```
For production-like local use, run `npm run build`. FastAPI then serves `frontend/dist` at `/`.

### Model files
- Python: `data/models/pose_landmarker_{lite,full,heavy}.task`. `rehabpose.pose.model_files` prints
  the official download URL and downloads only after explicit consent.
- Browser: the wasm runtime and the `.task` model are hosted under `frontend/public/` (copied at
  build time), so the production build has no runtime CDN dependency (NFR-09).

## 2. Docker (single container, CPU-only)
**Status: written but never built or run.** Building needs the Docker daemon and pulls base images,
which the owner has not approved yet. `docker compose config` validates the compose file, and tests
check the Dockerfile's structure and that `uv.lock` has CPU torch wheels for Linux x86_64 and arm64.

```bash
docker compose build
docker compose up
```
Then open http://127.0.0.1:8000.

- `docker/Dockerfile`, multi-stage:
  1. `node:24-slim` builds `frontend/dist`.
  2. `python:3.12-slim` installs the app with `uv sync --frozen --no-dev`. Torch comes from the CPU
     index, which the lock file guarantees ([ADR 0002](adr/0002-cpu-only-torch-lock.md)).
  3. It runs as the non-root user `rehabpose`. A healthcheck calls `GET /api/health`.
- `docker-compose.yml`: publishes `127.0.0.1:8000:8000` and mounts `./data:/app/data` (DB, models,
  uploads). The API has no login, so it binds to localhost only. Think about access control before
  publishing the port to a network.
- **Pose models are not in the image.** Put them in `data/models/`; the server serves
  `/models/pose_landmarker_{lite,full,heavy}.task` from there. Other files in that folder are not exposed.
- On Linux, `./data` must be writable by UID 1000.
- Containers on macOS have no access to the GPU/MPS. In-container inference (upload mode) is CPU
  only. Live mode is unaffected because pose runs in the browser.

## 3. Phone (same LAN)
Browsers allow camera access only on secure origins, so a phone needs HTTPS. `rehabpose serve` takes
`--ssl-certfile` and `--ssl-keyfile`, and the web app then uses `wss://` automatically. The exact commands
are in the [README](../README.md#use-a-phone-on-the-same-network).

**Status:** TLS and `wss://` were tested on the Mac with a self-signed certificate. The full flow on a
phone, with mkcert, has not been tested.

The API has no login, and `serve` logs a warning when bound beyond localhost. Use this only on a network
you trust. mkcert is a Homebrew package, so the owner is asked before it is installed.

## 4. Configuration
- `configs/app.yaml` + `REHABPOSE_*` env vars (e.g. `REHABPOSE_PORT=9000`), and
  `REHABPOSE_CONFIG=/path/app.yaml` for an alternative file.
- Exercise thresholds: `configs/exercises/*.yaml`. Per-user overrides live in the DB (`target` table).
- Data paths: `data/` (DB, models, uploads, interim), `mlruns/` (MLflow).

## 5. CI (GitHub Actions)
`.github/workflows/ci.yml`. **Status: the same commands pass locally on macOS; the workflow has never
run on GitHub, and nothing has been run on Linux.**

| Job | Runner | Steps |
|---|---|---|
| python | `macos-latest`, `ubuntu-latest` | `uv sync --frozen` → ruff → format check → mypy → `pytest -m "not perf" --cov=rehabpose.core --cov-fail-under=80` |
| frontend | `ubuntu-latest` | `npm ci` → lint → vitest → build |

There is no GPU or CUDA in CI. Tests use synthetic fixtures only; pose-model tests skip because models
are not committed. Tests marked `perf` (tight timing margin) run locally only. macOS runners install
Homebrew's `libomp` for LightGBM, and Linux runners install the OpenGL libraries OpenCV needs.

## 5b. Benchmarks
`rehabpose bench --synthetic --models lite,full,heavy --ws-replay`, or `--video <file>`, updates
[reports/benchmarks.md](../reports/benchmarks.md). Each model runs in its own process so peak memory is
its own. Run it with nothing else active and under `caffeinate -i`.

## 6. Release and versioning
- Phases are tagged `phase-N`. Releases use semantic versions (`v0.x`) after phase 10.
- Each session stores `model_version` (pose model variant + optional temporal model run id) for
  traceability.

## 7. Operational notes
- SQLite is single-user. There is one writer (the server process). Background jobs use a thread
  pool and write through the same engine.
- Logs go to stderr (rich in the CLI, plain under uvicorn).
- Backups mean copying `data/rehabpose.db`. Export/delete endpoints handle per-user data (FR-35).
