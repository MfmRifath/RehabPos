# RehabPose benchmarks

Last updated 2026-11-01T17:42:17+00:00. Regenerate with the commands below; numbers are from this machine only.

## Machine

- machine: arm64
- python: 3.12.15
- hw_model: Mac15,6
- chip: Apple M3 Pro
- memory_gb: 18
- macos: 26.6.2
- libraries: mediapipe 1.2.0, torch 2.15.0, numpy 2.5.4, opencv-python 5.0.0.93, fastapi 0.162.0, uvicorn 0.54.0, websockets 17.4

## Pose + engine (Python, CPU delegate)

`rehabpose bench --synthetic --models lite,full,heavy --ws-replay`

Video: rendered mannequin (SYNTHETIC rendered cartoon, not a human). Peak memory is for a whole process, per model.

| model | frames | detected | pose ms median / p95 | engine ms median / p95 | end-to-end FPS | peak MB |
|---|---|---|---|---|---|---|
| lite | 675 | 675 | 7.934 / 8.122 | 0.151 / 0.18 | 122.6 | 290.0 |
| full | 675 | 675 | 12.727 / 12.979 | 0.163 / 0.199 | 77.1 | 304.8 |
| heavy | 675 | 675 | 52.232 / 52.784 | 0.176 / 0.214 | 19.1 | 366.9 |

## Server latency (WebSocket replay, SYNTHETIC landmarks)

`rehabpose bench --synthetic --models lite,full,heavy --ws-replay`

1755 messages, one at a time, over localhost; 6 reps counted.

| measure | median ms | p95 ms |
|---|---|---|
| server processing | 0.168 | 0.193 |
| client round trip | 0.299 | 0.369 |

## Not measured here

- Live browser FPS on a real camera: NOT YET MEASURED (read from the app's HUD).
- Docker image build and run: NOT YET MEASURED.
- Accuracy on real people: NOT YET MEASURED.

---

RehabPose is a research and portfolio prototype. It is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or replace advice from a qualified healthcare professional. Stop exercising if you feel pain. Thresholds are engineering defaults, not clinical prescriptions.
