# 05 — Data plan

## 1. Principles
- **Nothing is downloaded automatically.** Each public dataset has a loader that reads from
  `data/raw/<name>/`. If the files are missing, it raises an error with download instructions.
  Download scripts in `scripts/download_*.sh` are written but only run after the owner's explicit
  approval.
- **Licences are accepted explicitly.** Loaders print the licence and citation, and require
  `--i-accept-license`.
- **Data is never committed.** `data/` is git-ignored. Only tiny synthetic fixtures (< 1 MB) live in
  `tests/fixtures/`.
- **Keypoints by default.** For our own recordings, raw video is kept only when consent covers it,
  and is stored locally.
- **Facts must be verified.** The dataset descriptions below come from the project spec. Licence
  terms marked *verify* must be checked against the dataset's own page before use, and the result
  recorded in this table.

## 2. Public datasets
| Key | Dataset | Sensor / modality | Content (as described in the spec) | Labels relevant to us | Licence | Access |
|---|---|---|---|---|---|---|
| `uco` | UCO Physical Rehabilitation (Sensors 2023) | 5 RGB views | 27 subjects, 16 exercises incl. seated shoulder flexion, horizontal openings, band external rotation | exercise class | verify | on request via github AVAuco/ucophyrehab |
| `uco` (++) | UCOPhyRehab++ (Scientific Data 2026) | derived modalities | extends UCO with physiotherapist 5-point scores, rep start/end frames | quality scores, rep boundaries, tempo rubric | verify | Zenodo |
| `rehab24_6` | REHAB24-6 | 2 RGB cams + mocap | 65 recordings, 10 subjects, 6 exercises, 1,072 segmented reps | correctness per rep | verify | Zenodo record 13305826 |
| `dib_upper` | Upper-limb stroke rehab exercise videos (Data in Brief 2026) | webcam + mobile | 491 videos, 4 exercises | complete / incomplete | CC BY 4.0 | Mendeley doi 10.17632/49h9dcwx5v.1 |
| `intellirehab` | IntelliRehabDS | Kinect | elbow flexion, shoulder flexion, abduction, forward elevation | correctness | CC BY 4.0 | Zenodo 10.5281/zenodo.4610859 |
| `uiprmd` | UI-PRMD | Vicon + Kinect | incl. shoulder abduction / extension / rotation / scaption | correct / incorrect | ODC PDDL | Zenodo |
| `kimore` | KIMORE | Kinect v2 | 78 subjects, 5 low-back exercises | clinical scores | check EULA | — |
| `trsp` | Toronto Rehab Stroke Pose | Kinect v2 | upper-body compensation frames | frame labels: none / lean-forward / shoulder-elevation / trunk-rotation | check terms | Kaggle derekdb/toronto-robot-stroke-posture-dataset |

### How each dataset is used
| Dataset | Rep counting | Angle accuracy | Form classification | Compensation (frame-level) | Pose pretraining |
|---|---|---|---|---|---|
| UCO / UCOPhyRehab++ | ✓ (rep boundaries) | — | ✓ (scores → labels) | — | ✓ |
| REHAB24-6 | ✓ | ✓ (mocap as reference) | ✓ | — | ✓ |
| Data in Brief upper-limb | — | — | ✓ (complete/incomplete ≈ INCOMPLETE_ROM) | — | — |
| IntelliRehabDS | — | — | ✓ | — | ✓ |
| UI-PRMD | — | ✓ (Vicon) | ✓ | — | ✓ |
| KIMORE | — | — | — | — | ✓ (low-back only) |
| Toronto Rehab Stroke Pose | — | — | — | ✓ (TRUNK_LEAN, SHOULDER_HIKE, TRUNK_ROTATION) | ✓ |

### Loaders (`rehabpose.data.loaders`, phase 4)
Each loader prints the licence and citation and requires `--i-accept-license`. Own data is the only
exception. If files are missing, the loader raises an error with download instructions. Each loader
returns `Sequence(dataset, subject, exercise, exercise_id, side, view, fps, source, joint_format,
frames | skeleton | video, labels, license, citation)`.

**The file layouts of all public datasets are ASSUMED** from the descriptions above. None has been
checked against real files yet (`layout_verified = False`, and the loader logs a warning). Each
assumed layout is a glob plus a filename regex. It can be overridden without code changes in
`configs/datasets/<name>.yaml` (`patterns`, `regex`, `fps`, `view`, `exercise_map`). After a
dataset is downloaded, verify the layout, update the override and this table, and only then map
native exercise codes to RehabPose ids.

| Dataset | Assumed layout | Payload |
|---|---|---|
| uco | `<subject>/<exercise>/<video>` | video (extract with MediaPipe) |
| rehab24_6 | `<subject>/<exercise>/<video>`; rep/correctness files not parsed yet | video |
| dib_upper | `<exercise>/<complete or incomplete>/<video>`; no subject ids, so subject-wise splits are impossible | video |
| intellirehab | `<subject>_<date>_<gesture>_<rep>_<correctness>_<position>.txt`, Kinect v2 rows | Kinect v2 25 joints |
| uiprmd | `<Segmented or Incorrect Segmented Movements>/Kinect/Positions/m<ex>_s<subj>_e<ep>_positions*.txt` | J×3 rows; joint order **not** assumed (`unknown_<J>`), not convertible yet |
| kimore | `<group>/[<subgroup>/]<subject>/Es<k>/Raw/JointPosition*.csv` | Kinect v2 25 joints |
| trsp | `<subject>/<sequence>.csv` with joint columns and an optional `label` column | Kinect v2 + frame labels |
| own | defined by this project (§8) | MediaPipe parquet + annotation JSON |

### Joint maps (`rehabpose.data.joint_maps`)
Skeleton datasets are mapped to a **13-joint upper-body subset** in MediaPipe indices: nose 0,
ears 7/8, shoulders 11/12, elbows 13/14, wrists 15/16, index fingers 19/20, hips 23/24. Unmapped
landmarks get visibility 0, and so do untracked source joints (zero or NaN). Coordinates are
converted to MediaPipe world axes (y down, +x = subject's left) and centred on the mid-hip:
Kinect camera space (x, y, z) → (−x, −y, z); a z-up mocap lab frame in mm → (x, −z, y)/1000.
Engine Frames are produced with a virtual orthographic camera (frontal or sagittal).

Approximations:
- **Kinect:** nose and both ears ← Head, so `hike_ratio` is approximate and horizontal ear
  positions are meaningless. Index finger ← HandTip.
- **Plug-in Gait:** ear ← mean(front, back head marker) per side. Wrist ← mean(WRA, WRB).
  Hip ← mean(ASI, PSI) per side, which is a crude proxy offset from the true hip joint centre.

`rehabpose data extract-all <name>` writes one parquet per sequence to `data/interim/<name>/`.
Videos go through MediaPipe with a fresh landmarker per video. Mappable skeletons are converted.
Formats with unknown joint order are skipped and counted as `unsupported`.

## 3. Own-recording protocol
**Goal:** a small, consented dataset of the five v1 exercises recorded with ordinary webcams. It serves
three purposes:
- the primary evaluation set (subject-wise),
- rep-level labels, including scripted compensations,
- manual angle annotations for angle accuracy.

| Item | Protocol |
|---|---|
| Participants | Healthy adult volunteers, IDs `P01..Pnn`. No patients in v1. Target size is decided with the owner before collection. |
| Hardware | Laptop webcam and/or phone, 30 fps, ≥ 720p. Camera at chest height, 2–3 m away. |
| Views | Sagittal (side-on) for flexion, elbow flexion and forward reach. Frontal for abduction and external rotation. Optionally a second simultaneous view. |
| Sides | Both arms, recorded as separate sets. |
| Sets per exercise | 1 × correct (8–10 reps at a comfortable controlled tempo), then scripted sets: incomplete ROM, too fast, and each applicable compensation (trunk lean, shoulder hike, elbow bend, shoulder drift, elbow away, trunk rotation). |
| Scripting | An instructor demonstrates each error. Participants exaggerate only within a comfortable range. No pain. |
| Environment | Vary lighting (bright/dim), background, clothing (fitted/loose) and sitting/standing, and log the conditions per set. |
| Rest | 1 s still at the start of every set (calibration) and rest between sets as needed. |
| Metadata | `subject, session, exercise, side, view, camera, fps, resolution, lighting, clothing, posture, scripted_error` |
| Storage | Raw video goes to `data/raw/own/videos/` (local only). Extracted landmarks go to `data/interim/own/*.parquet`. Labels go to `data/labels/own/*.json`. |

## 4. Consent outline
The full consent form is drafted with the owner before any recording. It must cover:
1. The purpose: a research/portfolio prototype, not a medical device or a clinical study.
2. What is recorded (video and/or keypoints), where it is stored (the owner's machine), and for how long.
3. Separate opt-ins for: (a) keypoints for model training, (b) keeping raw video, (c) showing frames
   or GIFs in public docs/README, (d) releasing keypoints publicly.
4. Pseudonymisation (`Pnn`). The mapping to names is kept offline and separately, or not kept at all.
5. The right to withdraw at any time, followed by deletion of all derived data, and a contact address.
6. Safety: stop on pain. No exercises beyond the participant's comfortable range.
7. Health screening: participants confirm they have no current upper-limb injury. Anyone with a
   condition is excluded from v1.

Ethics context and data-handling rules: [09_ethics_privacy.md](09_ethics_privacy.md).

## 5. Labeling guideline
Done with the labeling tool (`rehabpose label`, phase 4). Each video gets one JSON file.

- **Rep boundaries:** `start` = the first frame where the primary joint starts moving from rest.
  `peak` = the frame of maximum primary angle. `end` = the first frame back at rest. When unsure,
  pick the earlier frame for start and the later one for end.
- **Rep label:** `correct`, `incomplete_rom`, `too_fast` or `compensated`, using the same precedence
  as [04 §8](04_exercise_spec.md#8-rep-record-and-labels). The scripted error is a hint only: the
  annotator labels what is visible.
- **Compensation segments:** frame ranges with a code (`TRUNK_LEAN`, `SHOULDER_HIKE`, `ELBOW_BEND`,
  `SHOULDER_DRIFT`, `ELBOW_AWAY`, `TRUNK_ROTATION`, `PLANE_DRIFT`). Overlapping segments are allowed.
- **Angle annotation:** 3 clicks (proximal point, vertex, distal point) on sampled frames, about
  5 per rep, including the peak frame. The result is the reference for angle MAE and Bland–Altman.
- **Quality:** a second annotator relabels a random 20 % subset. We report agreement (Cohen's κ for
  labels, ICC/MAE for angles and boundaries). Disagreements are resolved by discussion, and the
  guideline is updated.

## 6. Annotation schema (v1)
Pydantic model `rehabpose.data.schema.Annotation` (`rehabpose data schema` prints the JSON Schema). Angle annotations are validated: `angle_deg` must match the angle computed from `points_px` within 0.5°. Example (illustrative values):
```json
{
  "schema_version": 1,
  "video_id": "P03_shoulder_flexion_R_sagittal_s1",
  "subject": "P03",
  "exercise_id": "shoulder_flexion",
  "side": "R",
  "view": "sagittal",
  "fps": 30.0,
  "annotator": "A1",
  "scripted_error": "TRUNK_LEAN",
  "conditions": {"lighting": "bright", "clothing": "fitted", "posture": "standing"},
  "reps": [
    {"idx": 0, "start_frame": 31, "peak_frame": 78, "end_frame": 130,
     "label": "compensated", "flags": ["TRUNK_LEAN"]}
  ],
  "compensation_segments": [
    {"code": "TRUNK_LEAN", "start_frame": 60, "end_frame": 101}
  ],
  "angle_annotations": [
    {"frame": 78, "measure": "upper_arm_trunk_angle",
     "points_px": [[640, 410], [655, 250], [610, 120]], "angle_deg": 141.2}
  ],
  "notes": ""
}
```

## 7. Splits
- **Unit of splitting = subject.** No subject appears in more than one of train/val/test
  (`rehabpose.data.splits`). A leakage test enforces this, including a property test over random
  subject counts. Sequences without subject ids (`unknown:*`) are rejected.
- `rehabpose data splits <name>` writes `configs/train/splits.yaml`. Loading a plan re-checks it
  for leakage.
- **Held-out test subjects:** about 20 % of own-data subjects, fixed before any model selection, with
  the IDs listed in `configs/train/splits.yaml`. They are evaluated once per reported model.
- **Model selection:** `GroupKFold` (k = 5) on the remaining subjects, or LOSO when there are fewer
  than 10 subjects.
- **Public datasets:** each is split by its own subject IDs. Subjects are never mixed across datasets.
  Cross-dataset results are reported separately (train on A, test on B).
- **Synthetic data** is used for tests and smoke runs only, and is always labelled "SYNTHETIC — not a
  result".

## 8. Directory layout
```
data/
  raw/<dataset>/          # as downloaded (manual), read-only
  interim/<dataset>/      # extracted landmarks / mapped skeletons (*.parquet)
  labels/<dataset>/       # annotation JSON
  models/                 # pose model files (*.task, *.onnx)
  rehabpose.db            # SQLite (app)
mlruns/                   # MLflow file store
```

## 9. Retention and deletion
- Own raw video is deleted after landmark extraction unless consent opt-in (b) was given.
- Withdrawal deletes raw video, landmarks, labels and derived artefacts for that subject. Models
  trained on that data are marked for retraining.
- App users can delete their data via the API (FR-35).
