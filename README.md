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

## Run the app (development)

```bash
uv run rehabpose serve                 # API + WebSocket on http://127.0.0.1:8000
cd frontend && npm ci && npm run dev   # UI on http://localhost:5173 (proxies /api and /ws)
```
Pose models must be in `data/models/` (`uv run rehabpose extract --download-model ...` fetches
them after consent). `npm run dev` copies them, together with MediaPipe's wasm, into
`frontend/public/`. Note that the dev server has no Content-Security-Policy. The production build
(`npm run build`, served by `rehabpose serve` at http://127.0.0.1:8000) blocks all third-party
requests, including MediaPipe's built-in telemetry (see docs/adr/0003).

## License

[Apache-2.0](LICENSE). The optional `[ablation]` extra installs Ultralytics (AGPL-3.0). It is never
imported by the core package.
