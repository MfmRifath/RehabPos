# RehabPose

Real-time, camera-only rehabilitation exercise coach: in-browser pose estimation, rep counting and
form feedback for upper-limb exercises. The camera video never leaves the device.

> **Disclaimer:** RehabPose is a research and portfolio prototype. It is not a medical device, has not
> been clinically validated, and must not be used to diagnose, treat, or replace advice from a
> qualified healthcare professional. Stop exercising if you feel pain. Thresholds are engineering
> defaults, not clinical prescriptions.

**Status:** Phase 0 (project bootstrap). See [PROGRESS.md](PROGRESS.md).

## Quickstart (development)

Requires an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run rehabpose --help
uv run pytest -q
```

## Run the app

Two options. Commands can be run from any folder inside the repo.

**Production-like (one terminal, recommended).** The build includes the privacy CSP.

```bash
cd ~/Projects/RehabPos/frontend
npm run build
cd ..
uv run rehabpose serve
```
Then open http://127.0.0.1:8000 in Chrome.

**Development (two terminals, hot reload).**

Terminal 1:
```bash
cd ~/Projects/RehabPos
uv run rehabpose serve
```
Terminal 2:
```bash
cd ~/Projects/RehabPos/frontend
npm run dev
```
Then open http://localhost:5173. Vite proxies `/api` and `/ws` to the server in terminal 1.

Pose models must be in `data/models/`; `uv run rehabpose extract --download-model ...` fetches
them after consent. `npm run build` / `npm run dev` copy them, together with MediaPipe's wasm,
into `frontend/public/`. The dev server has no Content-Security-Policy, so MediaPipe's built-in
telemetry is only blocked in the production build (see docs/adr/0003). Run `npm ci` only once,
after cloning or when `package-lock.json` changes.

## Run with Docker

Written but not yet built or run; see [docs/08](docs/08_deployment_plan.md). Pose models go in `data/models/`.

```bash
docker compose build
docker compose up
```

## Benchmarks

Measured numbers and the machine they came from are in [reports/benchmarks.md](reports/benchmarks.md).

## License

[Apache-2.0](LICENSE). The optional `[ablation]` extra installs Ultralytics (AGPL-3.0). It is never
imported by the core package.
