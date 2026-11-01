# 11 — Datasheet for the RehabPose own-recordings dataset

**This dataset does not exist yet: no one has been recorded.** This datasheet records the plan and the
questions it must answer, and every statistic reads NOT YET MEASURED until computed with
`rehabpose data stats own`. Update it when collection starts. The structure follows datasheets for
datasets.

## Motivation
- **Purpose:** to be the primary real-data evaluation set for rep counting, angle accuracy and form
  assessment, and to provide rep-level labels with scripted compensations. See
  [01_overview.md](01_overview.md).
- **Why a new dataset:** public datasets differ in camera setup, exercises or licence, and none is
  guaranteed to match a single-webcam, upper-limb use.
- **Creator:** the RehabPose project owner.
- **Funding:** none recorded.

## Composition
Planned content is from [05_data_plan.md](05_data_plan.md#3-own-recording-protocol).

| Item | Plan | Measured |
|---|---|---|
| Participants | Healthy adult volunteers, IDs `P01..Pnn`, no patients. Size to be decided before collection | NOT YET MEASURED |
| Exercises | The five v1 exercises, with both arms as separate sets | NOT YET MEASURED |
| Views | Sagittal or frontal per exercise, optionally a second view | NOT YET MEASURED |
| Sets | One correct set, plus scripted sets for each applicable error | NOT YET MEASURED |
| Label distribution | Per [the labeling guideline](05_data_plan.md#5-labeling-guideline) | NOT YET MEASURED |
| Conditions | Bright or dim lighting, varied background, fitted or loose clothing, sitting or standing | NOT YET MEASURED |
| Demographics | Only if participants agree. Fields are not yet decided | NOT YET MEASURED |

**Known gap:** scripted errors from healthy adults aren't real patient compensation. The dataset can't
support claims about patients.

**Sensitive content:** video shows faces, bodies and homes. Landmarks are less identifying but may still
reveal movement patterns or a health condition.

## Collection process
- **Hardware:** a laptop webcam and/or phone at 30 fps and at least 720p, chest height, 2–3 m away.
- **Procedure:** an instructor demonstrates each error, and participants exaggerate only within a
  comfortable range, and stop on any pain.
- **Dates and hardware actually used:** NOT YET MEASURED.

## Consent
The consent form is drafted before any recording, per the
[consent outline](05_data_plan.md#4-consent-outline). Participants confirm they have no current
upper-limb injury and can withdraw at any time. Keypoints for training, keeping raw video, showing frames
publicly and releasing keypoints are separate opt-ins. Opt-in counts: NOT YET MEASURED.

Whether ethics review is required depends on the institution and country, and has not been decided. **Get
that answer before recording anyone.**

## Preprocessing and labeling
- **Landmarks:** `rehabpose data extract-all own` produces landmark parquet with Python MediaPipe. The
  model variant must be recorded here. Used so far: NOT YET MEASURED.
- **Labels:** `rehabpose label`, following the
  [guideline](05_data_plan.md#5-labeling-guideline) and the
  [annotation schema](05_data_plan.md#6-annotation-schema-v1).
- **Inter-annotator agreement:** NOT YET MEASURED.
- **Raw data:** raw video is kept only where consent covers it.

## Uses
- **Intended:** evaluating and training RehabPose models, with subject-wise splits only.
- **Not suitable for:** clinical conclusions, patient models, identifying people or inferring health
  conditions. See [09_ethics_privacy.md](09_ethics_privacy.md#2-intended-use-and-out-of-scope-use).

## Distribution
Not distributed by default, and data is never committed to this repository. Any release would contain
only keypoints from participants who opted in. Licence if released: not decided.

## Maintenance
- **Maintainer:** the project owner. A contact address will be added before any release.
- **Withdrawal:** a participant's data is deleted on request, per
  [the retention rules](05_data_plan.md#9-retention-and-deletion).
- **Versioning:** to be decided if a release happens. Each version would need its own statistics here.
