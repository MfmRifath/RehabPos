# ADR 0003 — Same-origin Content-Security-Policy for the production frontend

- **Status:** accepted
- **Date:** 2026-11-01
- **Deciders:** project owner, Claude Code

## Context
RehabPose promises that, in production, nothing leaves the user's machine: no video, no
telemetry, no analytics (docs/09 §4, NFR-05, NFR-09). The phase 6 Playwright smoke test checks
for requests to other origins. It caught **`@mediapipe/tasks-vision` 1.2.0 POSTing usage logs to
`https://odml.pa.googleapis.com/v1/log`**. Inspecting the bundle showed a logger class that is
built internally with an API key and flushes on a 60 s timer and on close. The library's public
API offers no option to turn it off.

## Decision
The production build carries a Content-Security-Policy `<meta>` tag. It is injected by a Vite
build plugin (`frontend/vite.config.ts`); the dev server has none because it needs inline scripts
for HMR. The policy:

```
default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self';
img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline';
worker-src 'self' blob:; object-src 'none'; base-uri 'self'; form-action 'self'
```

`connect-src 'self'` makes the browser refuse every fetch, XHR or WebSocket to another origin,
including MediaPipe's telemetry. `'wasm-unsafe-eval'` is the minimum MediaPipe's wasm needs.

## Verification (phase 6)
- `e2e/smoke.spec.ts` checks that the policy is present and that the exact telemetry POST is
  refused (`fetch` rejects). It also asserts that no request goes to another origin during the full
  UI flow, and that the pose model still loads (the GPU delegate worked in headless Chrome 156).
- `e2e/real-backend.spec.ts` checks the same against `rehabpose serve` serving `frontend/dist`.

## Consequences
- The privacy promise holds even if a dependency adds network calls. They fail closed and show up
  as CSP console errors.
- The dev server (`npm run dev`) has no CSP, so MediaPipe telemetry can be sent during
  development. The README says so.
- Any future intentional third-party resource needs an explicit policy change and an ADR.
- Inline styles stay allowed (`'unsafe-inline'` for style-src only), because React style
  attributes need them.
