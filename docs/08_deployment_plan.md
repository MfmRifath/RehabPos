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
- `docker/Dockerfile`, multi-stage:
  1. `node:24` builds `frontend/dist`.
  2. A `python:3.12-slim` runtime installs the app with uv (`uv sync --frozen --no-dev`). Torch comes
     from the CPU index, which the lock file guarantees ([ADR 0002](adr/0002-cpu-only-torch-lock.md)).
  3. It runs as a non-root user. A healthcheck calls `GET /api/health`.
- `docker-compose.yml`: port `8000:8000`, volume `./data:/app/data` (DB, models, uploads).
- Containers on macOS have no access to the GPU/MPS. In-container inference (upload mode) is CPU
  only. Live mode is unaffected because pose runs in the browser.
- Before pulling large base images, the owner is asked.

## 3. Phone (same LAN)
Browsers allow camera access only on secure origins, so a phone needs HTTPS.

1. Install mkcert. This is a Homebrew package, so the owner is asked before installing it. Then run
   `mkcert -install`.
2. Run `mkcert <LAN-IP> localhost 127.0.0.1` to create a cert and key in a git-ignored folder.
3. Serve with TLS, e.g.
   `uv run rehabpose serve --host 0.0.0.0 --ssl-certfile … --ssl-keyfile …` (phase 5 adds the flags),
   or Vite `server.https`.
4. Install the mkcert root CA on the phone (iOS: Settings → General → About → Certificate Trust
   Settings).
5. Open `https://<LAN-IP>:8000` on the phone.

Exact commands are written in the README in phase 10.

## 4. Configuration
- `configs/app.yaml` + `REHABPOSE_*` env vars (e.g. `REHABPOSE_PORT=9000`), and
  `REHABPOSE_CONFIG=/path/app.yaml` for an alternative file.
- Exercise thresholds: `configs/exercises/*.yaml`. Per-user overrides live in the DB (`target` table).
- Data paths: `data/` (DB, models, uploads, interim), `mlruns/` (MLflow).

## 5. CI (GitHub Actions, phase 9)
| Job | Runner | Steps |
|---|---|---|
| python | `macos-latest`, `ubuntu-latest` | `uv sync --frozen` → ruff → mypy → `pytest --cov` (fail-under 80 for core) |
| frontend | `ubuntu-latest` | `npm ci` → lint → vitest → build |

There is no GPU or CUDA in CI. Tests use synthetic fixtures only.

## 6. Release and versioning
- Phases are tagged `phase-N`. Releases use semantic versions (`v0.x`) after phase 10.
- Each session stores `model_version` (pose model variant + optional temporal model run id) for
  traceability.

## 7. Operational notes
- SQLite is single-user. There is one writer (the server process). Background jobs use a thread
  pool and write through the same engine.
- Logs go to stderr (rich in the CLI, plain under uvicorn).
- Backups mean copying `data/rehabpose.db`. Export/delete endpoints handle per-user data (FR-35).
