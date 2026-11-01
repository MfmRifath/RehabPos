# Portfolio notes

CV bullets for RehabPose, using measured numbers only. Pick the ones that fit the job.

Keep the word "synthetic" wherever an accuracy number appears. Those numbers show the pipeline works, not
that it's accurate on people. Saying so up front will read as rigor, and an interviewer would find out
otherwise.

## CV bullets

**Full stack and real-time**
- Built a camera-only exercise coach (React, TypeScript, FastAPI) that runs MediaPipe pose estimation in
  the browser and streams only landmarks to the server, so video never leaves the device.
- Designed a versioned WebSocket protocol with latest-frame backpressure, measuring 0.168 ms median and
  0.193 ms p95 server processing per message in a 1,755-message synthetic replay on an M3 Pro.
- Benchmarked three pose models on an M3 Pro, from 122.6 FPS (lite) to 19.1 FPS (heavy) end to end on the
  CPU, to guide the speed and accuracy trade-off on a rendered test video.

**Signal processing and rules**
- Implemented a tested biomechanics engine in Python: One Euro smoothing, joint angles, a hysteresis rep
  state machine, YAML-configured form rules and DTW similarity scoring.
- Ran ablations that challenged my own design: smoothing raised synthetic shoulder angle error from 0.77° to
  2.26°, so I logged it as an open issue instead of assuming the filter helped.

**Machine learning**
- Built a reproducible training pipeline on Apple Silicon (PyTorch MPS, LightGBM, MLflow) comparing six
  architectures with subject-wise cross-validation, held-out test subjects and ONNX export with parity checks.
- Evaluated 10 model configurations on a synthetic skeleton generator and reported honest weaknesses, such
  as a Transformer ranging from 0.608 to 0.940 macro-F1 across seeds, and a rule baseline at 0.965 that only
  LightGBM consistently beat.

**Engineering quality and responsible practice**
- Wrote 499 Python and 34 frontend tests, with over 98 % coverage of the core engine, plus a check that
  fails if README or model card metrics stop matching the measured results file.
- Wrote a model card and datasheet stating what hasn't been measured, including accuracy on real people, and
  checked each dataset licence against its source, which caught a non-commercial restriction.

## What not to claim
- "Accurate" or "validated": nothing has been tested on real people.
- "Clinical", "medical" or "for patients": this is not a medical device, and no patients were involved.
- "Deployed", "Dockerized" or "CI/CD": those files exist, but none has been built or run.
- "Works on phones": the phone flow hasn't been tested.
- Any real-time browser FPS: it hasn't been measured.

## Sources
- Pipeline numbers: [reports/results_summary.json](../reports/results_summary.json).
- Speed: [reports/benchmarks.md](../reports/benchmarks.md).
- Tests and coverage: run `uv run pytest -q --cov=rehabpose.core`. A test checks the decimal numbers in the
  bullets above against the results file.

## Before sharing
- Record a demo GIF with consent. The README has a placeholder.
- Push the repository, and check CI actually passes before linking it.
