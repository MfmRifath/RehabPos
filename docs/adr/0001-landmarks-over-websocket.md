# ADR 0001 — Run pose in the browser and send only landmarks over WebSocket

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** project owner, Claude Code

## Context
Live coaching needs pose estimation on every frame, feedback within a fraction of a second, and as
little privacy exposure as possible. The options considered:

1. **Browser pose → landmarks over WS → Python engine** (this decision).
2. Browser streams video/JPEG frames → server-side pose + engine.
3. Everything in the browser (pose + rules in TypeScript), with the backend only for storage.

MediaPipe Tasks Vision runs in the browser with WebGL/WebGPU acceleration. The Python Tasks API on
macOS runs on CPU, since the Python GPU delegate gives no reliable speedup there. Docker on macOS has
no GPU access at all.

## Decision
The browser runs MediaPipe Pose Landmarker (VIDEO mode, GPU delegate with CPU fallback) and sends only
33 image landmarks + 33 world landmarks per frame, as WS protocol v1 messages
([03 §7](../03_architecture.md#7-websocket-protocol-v1)), to FastAPI. The single Python
`SessionEngine` performs smoothing, angles, rep counting, rules and optional ML, so live and upload
modes share identical logic. Server-side frame streaming (`/ws/frames/{sid}`) exists only as an
optional path for pose-backend ablations.

## Consequences
**Positive**
- Privacy: no video leaves the device in live mode (NFR-05).
- Bandwidth: one message carries 66 × 4 floats (a few KB of JSON), compared with tens of KB per JPEG frame.
- Pose runs on the client GPU. The server works on CPU and in Docker.
- One engine implementation (Python) is tested once and reused for uploads and evaluation.

**Negative / risks**
- Browser and Python MediaPipe builds may differ slightly, so live and upload results are not
  bit-identical. Mitigation: the same model variant and thresholds on both, and cross-check tests on
  recorded landmark JSONL.
- The round-trip adds network latency to cue display. This is acceptable because cues are debounced
  (≥ 0.5 s) anyway. Latency is measured in the HUD.
- The client is trusted to send honest landmarks. That is acceptable for a local, single-user prototype.
- Alternative 3 (all-TypeScript) would remove the server round-trip but duplicate the engine logic.
  We may revisit it with a WASM/TS port later, which would need a new ADR.
