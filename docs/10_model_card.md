# 10 — Model card (stub)

Status: **stub**. This card is completed in phase 10, using only values from `reports/eval/*.md`
(with MLflow run ids). Until then, every metric reads NOT YET MEASURED.

> RehabPose is a research and portfolio prototype. It is not a medical device, has not been
> clinically validated, and must not be used to diagnose, treat, or replace advice from a qualified
> healthcare professional. Stop exercising if you feel pain. Thresholds are engineering defaults, not
> clinical prescriptions.

## Model details
- Name / version: NOT YET MEASURED
- Architecture: NOT YET MEASURED (candidates in [06_ml_eval_plan.md](06_ml_eval_plan.md#3-models-rehabposemlmodels))
- Input: causal windows of upper-body landmarks and angles. Output: form labels / compensation probabilities.
- Training run ids (MLflow): NOT YET MEASURED

## Intended use
See [09_ethics_privacy.md](09_ethics_privacy.md#2-intended-use-and-out-of-scope-use).

## Out-of-scope use
See [09_ethics_privacy.md](09_ethics_privacy.md#2-intended-use-and-out-of-scope-use).

## Training data
NOT YET MEASURED. See [05_data_plan.md](05_data_plan.md) for candidate datasets.

## Evaluation data
NOT YET MEASURED.

## Metrics (with 95 % CIs)
| Metric | Target | Measured | Source |
|---|---|---|---|
| Form macro-F1 (subject-wise, own data) | ≥ 0.85 | NOT YET MEASURED | — |
| Rep-count MAE | ≤ 0.5 | NOT YET MEASURED | — |
| Shoulder angle MAE | ≤ 8° | NOT YET MEASURED | — |
| Elbow angle MAE | ≤ 10° | NOT YET MEASURED | — |
| Inference per window | ≤ 5 ms | NOT YET MEASURED | — |

## Subgroup results
NOT YET MEASURED.

## Limitations
NOT YET MEASURED. Known risks are listed in [04_exercise_spec.md](04_exercise_spec.md#11-known-limitations).

## Ethical considerations
See [09_ethics_privacy.md](09_ethics_privacy.md).
