# 06 — ML and evaluation plan

The rule engine ([04_exercise_spec.md](04_exercise_spec.md)) is the baseline and is always available.
ML is an **optional** layer that must beat or complement the rules under subject-wise evaluation
before the engine enables it.

## 1. Tasks
| Task | Unit | Output | Baselines |
|---|---|---|---|
| T1 Rep form classification | one segmented rep | `correct` / `incomplete_rom` / `too_fast` / `compensated` | rule labels; LightGBM on rep features |
| T2 Streaming compensation detection | causal window (30–60 frames, stride 5) | per-code probabilities (multi-label) | rule flags |
| T3 Rep counting | set | count | FSM (rules) |
| T4 Angle accuracy | frame | measure in degrees | n/a (measured vs manual annotation / mocap) |

T3 and T4 evaluate the deterministic pipeline. T1 and T2 evaluate learned models.

## 2. Features and preprocessing (`rehabpose.ml.transforms`)
- Centre on mid-hip and scale by shoulder width. Optionally rotation-align (shoulder line horizontal).
- Features: 13 upper-body joints xy (+ z when 3D), engine angles, angular velocities, and visibility.
- Augmentations (train only): jitter, scale ±10 %, rotation ±10°, time-warp ±20 %, joint dropout, and
  mirroring with L/R side swap.
- Rep-level samples are resampled to 64 steps. Streaming windows are 30–60 frames with stride 5, and
  causal only (no future frames).

## 3. Models (`rehabpose.ml.models`)
| Model | Use | Notes |
|---|---|---|
| LightGBM (`lgbm.py`) | T1 baseline | per-rep summary features: peak, duration, velocities, rule-measure stats |
| TCN (`tcn.py`) | T1/T2 | causal dilated convolutions, small (target < 1 M params) |
| LSTM (`lstm.py`) | T1/T2 | causal LSTM. A BiLSTM variant is for offline T1 only. |
| Transformer (`transformer.py`) | T1/T2 | 2–4 layers, d = 64, causal mask |
| ST-GCN (`stgcn.py`) | optional | small graph conv over the 13-joint graph |

## 4. Training protocol (`rehabpose.ml.train`)
- Configs live in `configs/train/*.yaml` (model, features, window, augmentation, optimiser, seeds).
- AdamW, cosine schedule, early stopping on fold-validation **macro-F1**, class weights for imbalance.
- Seeds `[0, 1, 2]`. We report mean ± std across seeds and folds.
- Device order: `mps` if available, else `cpu` (`rehabpose.settings.torch_device`). No CUDA.
- MLflow (local file store `mlruns/`) logs params, per-fold metrics, confusion matrices, model
  artefacts and git commit.

## 5. Evaluation protocol (`rehabpose.ml.evaluate`)
- **Subject-wise** splits only (GroupKFold / LOSO plus held-out test subjects). See
  [05_data_plan.md](05_data_plan.md#7-splits). A leakage test runs before every evaluation.
- Metrics:
  - **T1/T2:** macro-F1 (primary), per-class F1, AUROC (one-vs-rest), confusion matrices.
  - **T3:** rep-count MAE per set, and exact-count rate.
  - **T4:** MAE and RMSE in degrees per measure, plus Bland–Altman (bias, 95 % limits of agreement)
    against manual annotations or mocap.
  - **Latency:** per-window inference time on MPS and CPU (median and p95). We report the exact device.
- **Confidence intervals:** subject-level bootstrap (resample subjects, 1,000 draws), 95 % CI.
- **Subgroups** (when metadata allows): view, lighting, clothing, posture, side.
- Output: `reports/eval/<run>.md` + figures, citing MLflow run ids and config hashes.

## 6. Targets (from the spec; not results)
| Metric | Target |
|---|---|
| T1 macro-F1, subject-wise, own data | ≥ 0.85 |
| T3 rep-count MAE | ≤ 0.5 |
| T4 shoulder angle MAE | ≤ 8° |
| T4 elbow angle MAE | ≤ 10° |
| Inference per window | ≤ 5 ms |

Phase 7 reports measured values against these targets honestly, including misses. Results on
synthetic data are always labelled "SYNTHETIC — not a result".

## 7. Online use (`rehabpose.ml.online`)
- A streaming wrapper keeps a ring buffer of features and runs the model every `stride` frames.
- Model flags carry `source: "model"` and confidence = probability.
- Fusion with rules is configurable: `rule OR (p > threshold)`, with a per-code threshold chosen on
  validation folds only.
- Models are exported to TorchScript and ONNX (`rehabpose.ml.export`) with a parity test
  (`|diff| < 1e-4` on fixtures).

## 8. Ablations
| Ablation | Variants |
|---|---|
| Pose backend | MediaPipe lite / full / heavy, RTMPose via rtmlib (`[rtm]`), YOLO-pose (`[ablation]`, AGPL, optional) |
| Angle space | 2D image angles vs 3D world angles |
| Smoothing | One Euro on/off, parameter sweep |
| Window length | 30 / 45 / 60 frames |
| Pretraining | with / without pretraining on public skeleton datasets |

## 9. Reporting rules
- Every number in docs comes from `reports/eval/*.md`, which cites MLflow run ids and the command used.
- No cherry-picking: we report all folds and seeds, plus the held-out test set once per final model.
- The model card ([10_model_card.md](10_model_card.md)) is filled only from these reports.
