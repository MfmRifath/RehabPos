# 09 — Ethics, safety and privacy

## 1. Disclaimer (verbatim, shown in README, UI first-launch modal and every report)
> RehabPose is a research and portfolio prototype. It is not a medical device, has not been
> clinically validated, and must not be used to diagnose, treat, or replace advice from a qualified
> healthcare professional. Stop exercising if you feel pain. Thresholds are engineering defaults, not
> clinical prescriptions.

In code it is `rehabpose.DISCLAIMER`. Tests check that README, this document and the reports contain it.

## 2. Intended use and out-of-scope use
| Intended | Out of scope |
|---|---|
| Demonstrating real-time pose-based exercise feedback | Diagnosis, treatment decisions, prescribing exercise |
| Research on camera-based rep counting and form assessment | Use without a clinician's guidance for patients with acute injury, post-op restrictions or neurological conditions |
| Healthy-volunteer practice and portfolio demos | Safety-critical or unsupervised clinical monitoring |
| A therapist exploring objective home-practice data with caveats | Insurance, employment or legal decisions |

## 3. Safety design
- **One cue at a time**, short and non-alarming. The UI never says "wrong" or "injury".
- The **stop-on-pain** message is part of the disclaimer modal and the session start screen.
- **Conservative defaults:** INCOMPLETE_ROM is informational. The app never encourages pushing past
  the target, and target ROM is set by the user or therapist.
- **Unknown over wrong:** when visibility or view are inadequate, the system says so and stays silent
  on form (FR-04).
- **Experimental features are labelled** (external rotation, PLANE_DRIFT), and their measures are
  excluded from accuracy claims.

## 4. Privacy by design
- **Live mode is landmarks-only.** The video stays in the browser and only 33 + 33 landmark tuples
  per frame are sent (see [ADR 0001](adr/0001-landmarks-over-websocket.md)).
- **Local-first:** the server runs on the user's own machine or container. There is no third-party
  analytics, telemetry or CDN at runtime in the production build.
- **Pseudonymous users:** `user.id` is random, and `display_name` is optional and free text. No email
  or real-name requirement.
- **Upload mode:** uploaded videos are stored under `data/` only while the job runs. The annotated
  output is kept until the user deletes the session. A retention setting arrives in phase 8.
- **Data subject rights:** export (JSON) and delete-with-cascade endpoints (FR-35).
- **Logs** contain no landmarks or video, only session ids, codes and timings.

## 5. Research data ethics
- Own recordings follow the consent outline in [05_data_plan.md](05_data_plan.md#4-consent-outline):
  separate opt-ins for training, raw-video retention, public display and public release.
- No patient data in v1. Healthy volunteers only.
- Faces: keypoints-only storage is the default. Any frame or GIF shown publicly requires opt-in (c),
  or a synthetic/self recording by the owner.
- Public datasets are used only under their licences. Loaders print the licence and citation, and
  require explicit acceptance. EULA-restricted data (e.g. KIMORE) is never redistributed, and neither
  are models trained on it if the EULA forbids that.

## 6. Fairness and limitations
Pose estimators and our rules may perform differently across:
- body types and BMI, skin tone and lighting, loose clothing, and assistive devices (wheelchairs,
  slings),
- age, mobility level, and views other than the prescribed one.

Mitigations: log recording conditions, report subgroup metrics where data allows
([06_ml_eval_plan.md](06_ml_eval_plan.md#5-evaluation-protocol-rehabposemlevaluate)), state
limitations in the model card ([10_model_card.md](10_model_card.md)), and fall back to "unknown"
rather than guessing.

## 7. Licensing and third-party code
- The core repository is Apache-2.0 ([LICENSE](../LICENSE)).
- Ultralytics (AGPL-3.0) is only in the optional `[ablation]` extra and in `pose/yolo_pose.py` behind
  a lazy import. The core package never imports it. Distributing a build that includes it triggers
  AGPL obligations, so the default Docker image excludes it.
- MediaPipe models and runtime are used under their published terms, which are checked when the
  model download helper is written (phase 2).
- Dataset licences are tracked in [05_data_plan.md](05_data_plan.md#2-public-datasets).

## 8. Honest reporting
No fabricated results: every number traces to a command and an output file. Unmeasured items read
**NOT YET MEASURED** (see [07_testing_plan.md](07_testing_plan.md#7-review-checklist-for-numbers-nfr-14)).
