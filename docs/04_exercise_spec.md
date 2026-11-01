# 04 — Exercise and biomechanics specification

Every threshold here is a **default** stored in `configs/exercises/<id>.yaml`. A therapist can override
it per user (FR-33). These are engineering defaults, **not clinical prescriptions**. Angles are in
degrees, times in seconds, and distances in normalized units (shoulder widths) unless noted.

## 1. Landmark schema
MediaPipe Pose Landmarker returns 33 landmarks per person:

- `lm[33, 4]`: (x, y, z, visibility). x and y are normalized to the image ([0, 1]). z is relative depth
  (same scale as x, origin at the hips).
- `world[33, 4]`: (x, y, z, visibility) in metres, hip-centred.

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | nose | 7 / 8 | left / right ear |
| 11 / 12 | left / right shoulder | 13 / 14 | left / right elbow |
| 15 / 16 | left / right wrist | 23 / 24 | left / right hip |

Side selection: `L` uses the odd indices (7, 11, 13, 15, 23) and `R` uses the even indices (8, 12, 14,
16, 24). "Left" is the subject's anatomical left, as MediaPipe defines it.

**2D angles use pixel-scaled coordinates** `(x·w, y·h)`, so that non-square frames do not distort
angles. **3D angles** use `world`.

Derived points: `mid_shoulder = (lm11 + lm12)/2`, `mid_hip = (lm23 + lm24)/2`.

## 2. Quality gates and calibration
- **Visibility gate:** if any landmark required by a measure has visibility < **0.6**, that measure is
  `unknown` (`null` in the WS payload). Rules depending on it do not fire, so no corrective cue is
  given. A setup cue is given instead if the condition persists.
- **View check:** the frontal/sagittal heuristic uses the shoulder-width/torso-length ratio. A wide
  ratio means frontal, a narrow one means sagittal. A mismatch with the exercise's `view` sets
  `quality.view_ok = false`. Rules marked view-dependent then become `unknown`.
- **Calibration:** during the first **1.0 s** of a set, the user is at rest. The engine averages:
  - `baseline_trunk_tilt` (deg)
  - `baseline_ear_shoulder` (vertical ear–shoulder distance on the exercising side, px)
  - `baseline_shoulder_width` (px, distance between landmarks 11 and 12)

  If visibility fails during calibration, the engine extends the window, up to 3 s, then reports
  `setup_status` with hints.
- **Normalization unit:** shoulder width, the distance between landmarks 11 and 12. It is used for all
  distance thresholds and for One Euro speed normalization. In side-on (sagittal) 2D views the
  shoulders overlap and the projected width collapses towards 0, so the implementation
  (`rehabpose.core.angles.body_scale`) floors it at 0.7 × torso length (mid-hip to mid-shoulder).
  0.7 is roughly the adult shoulder-width to torso-length ratio. In frontal views the floor is
  normally inactive.

## 3. Measures
Notation: `S`, `E`, `W`, `H`, `Ear` = shoulder, elbow, wrist, hip, ear on the exercising side.
`angle(u, v)` is the unsigned angle between vectors u and v, in [0°, 180°].

| Measure | Definition | Range / meaning |
|---|---|---|
| `upper_arm_trunk_angle` | `angle(E − S, H − S)` | 0 = arm at side, 180 = overhead |
| `elbow_interior` | `angle(S − E, W − E)` | 180 = straight arm |
| `elbow_flexion` | `180 − elbow_interior` | 0 = straight, larger = more flexed |
| `trunk_tilt` | `angle(mid_shoulder − mid_hip, up)` with `up = (0, −1)` in image coords | 0 = upright |
| `hike_ratio` | `(S_y − Ear_y) / shoulder_width`, compared to baseline | drops when the shoulder hikes toward the ear |
| `shoulder_width_ratio` | `shoulder_width / baseline_shoulder_width` | < 1 when the trunk rotates (frontal view) |
| `forearm_rotation` (exp.) | angle of `W − E` about the upper-arm axis, from `world` | external rotation (ER exercise) |
| `reach_ratio` (opt.) | forward displacement of `W` from its rest position / arm length (`norm(S−E) + norm(E−W)`) | forward reach |

For shoulder abduction, `upper_arm_trunk_angle` is computed from 2D image coordinates in a frontal
view, which is a projection onto the frontal plane. For flexion it is computed in a sagittal view, a
projection onto the sagittal plane. Phase 7 ablations compare 2D against 3D (world) angles.

## 4. Rep detection (hysteresis FSM)
```mermaid
stateDiagram-v2
    [*] --> rest
    rest --> ascending : primary rises above start_below
    ascending --> rest : falls back below start_below (aborted)
    ascending --> peak : primary above target minus peak_above_offset
    peak --> descending : primary starts decreasing
    descending --> rest : primary below return_below (rep candidate)
```

A rep candidate becomes a rep only if:
- duration (t_end − t_start) ≥ **0.8 s** (global minimum), and
- peak ≥ `start_below + 20°`.

Partial reps, where the angle rises but never reaches the peak threshold, are recorded as
`incomplete_rom` if the peak ≥ `start_below + 20°`. Otherwise they are ignored. The FSM runs on the
**smoothed** primary angle. Unknown frames hold the current state. If more than 0.5 s of frames is
unknown, the FSM resets to `rest` and the filter resets.

Each exercise YAML defines `fsm: {start_below, peak_above_offset, return_below}`. Flexion uses the
spec's defaults (start < 30, peak > target − 15, return < 40). The other exercises use the same values
until phase 3 tunes them on synthetic data; those values are marked *provisional* in the YAML.

## 5. Exercises

### 5.1 `shoulder_flexion`
| Field | Default |
|---|---|
| view | sagittal |
| primary | `upper_arm_trunk_angle` |
| target | 150° |
| fsm | start_below 30, peak_above_offset 15, return_below 40 |
| min_rep_s | 2.0 |
| max_peak_vel_dps | 90 |

| Rule | Condition | Notes |
|---|---|---|
| ELBOW_BEND | `elbow_interior < 160` for > 0.3 s | |
| TRUNK_LEAN | `abs(trunk_tilt − baseline) > 10` | |
| INCOMPLETE_ROM | rep peak < 0.9 × target | evaluated at rep end |
| TOO_FAST | duration < 2.0 s **or** peak velocity > 90 °/s | evaluated at rep end |
| SHOULDER_HIKE | — | always `unknown` in sagittal view |

### 5.2 `shoulder_abduction`
| Field | Default |
|---|---|
| view | frontal |
| primary | `upper_arm_trunk_angle` (frontal projection) |
| target | 130° |
| fsm | start_below 30, peak_above_offset 15, return_below 40 (provisional) |
| min_rep_s | 2.0 |
| max_peak_vel_dps | 90 |

| Rule | Condition | Notes |
|---|---|---|
| TRUNK_LEAN | `abs(trunk_tilt − baseline) > 8` | lateral lean |
| SHOULDER_HIKE | `hike_ratio` drop > 15 % vs baseline | |
| ELBOW_BEND | `elbow_interior < 160` for > 0.3 s | |
| PLANE_DRIFT | world `abs(W_z − S_z) > 0.15 m` | experimental; monocular depth is unreliable |
| INCOMPLETE_ROM | peak < 0.9 × target | |
| TOO_FAST | duration < 2.0 s or peak velocity > 90 °/s | |

### 5.3 `elbow_flexion`
| Field | Default |
|---|---|
| view | sagittal (frontal allowed with a setup warning) |
| primary | `elbow_flexion` |
| target | 130° |
| fsm | start_below 30, peak_above_offset 15, return_below 40 (provisional) |
| min_rep_s | 1.5 |
| max_peak_vel_dps | 90 |

| Rule | Condition |
|---|---|
| SHOULDER_DRIFT | `upper_arm_trunk_angle > 20` (upper arm should stay at the side) |
| TRUNK_LEAN | `abs(trunk_tilt − baseline) > 10` |
| INCOMPLETE_ROM | peak < 0.9 × target |
| TOO_FAST | duration < 1.5 s or peak velocity > 90 °/s |

### 5.4 `shoulder_external_rotation` (experimental)
| Field | Default |
|---|---|
| view | frontal |
| primary | `forearm_rotation` (world landmarks) |
| target | set in YAML, provisional until validated |

| Rule | Condition |
|---|---|
| ELBOW_AWAY | `abs(E_x − H_x) > 0.25 × shoulder_width` (elbow should stay tucked at the side) |
| TRUNK_ROTATION | `shoulder_width_ratio < 0.85` (shrink > 15 %) |

Monocular estimates of forearm rotation are known to be weak. The exercise is labelled experimental in
the UI, and its measures are excluded from accuracy targets until they are validated.

### 5.5 `forward_reach` (optional)
| Field | Default |
|---|---|
| view | sagittal |
| primary | `reach_ratio` (FSM thresholds in ratio units, provisional) |

| Rule | Condition |
|---|---|
| TRUNK_LEAN | `abs(trunk_tilt − baseline) > 15` |
| SHOULDER_HIKE | as in abduction; `unknown` in sagittal view |
| TRUNK_ROTATION | as in external rotation; needs frontal cues, so `unknown` in pure sagittal view |

## 6. Rules, cues and priority
- **Severity:** `info` (setup hints), `warn` (compensation), `error` (unsafe or invalid setup).
- **Confidence:** for rules, this is the minimum visibility of the involved landmarks, scaled to
  [0, 1]. For the model, it is the predicted probability.
- **Unknown handling:** a rule whose inputs are `unknown` or view-dependent in the wrong view yields
  no flag and no cue.
- **Priority** (highest first): visibility/setup > `TRUNK_*` > `SHOULDER_HIKE` > `ELBOW_*` (incl.
  `SHOULDER_DRIFT`) > `TOO_FAST` > `INCOMPLETE_ROM`. `PLANE_DRIFT` ranks with `SHOULDER_*`.
- **Display:** at most one cue at a time.
- **Debounce:** a cue's condition must persist **≥ 0.5 s** before it fires.
- **TTS rate limit:** at most **1 spoken cue per 3 s**. Captions follow the same cue.
- **Cue texts:** each YAML contains `cue_texts: {CODE: "short imperative"}`, e.g.
  `TRUNK_LEAN: "Keep your trunk upright"`.

## 7. Tempo
- Angular velocity uses causal finite differences on the smoothed primary angle (°/s).
- About **45 °/s** is considered a controlled tempo. This reference comes from the UCOPhyRehab++
  scoring rubric (see [05_data_plan.md](05_data_plan.md)). It is a reference only, not a threshold.
- `TOO_FAST` defaults: duration below `min_rep_s` or peak velocity above `max_peak_vel_dps` (90 °/s).

## 8. Rep record and labels
| Field | Unit | Description |
|---|---|---|
| idx | — | 0-based rep index in the set |
| t_start, t_peak, t_end | s | from frame timestamps |
| peak_angle | deg | max of the smoothed primary measure |
| duration_s | s | t_end − t_start |
| peak_vel_dps | °/s | max abs angular velocity during the rep |
| flags | list | codes raised during the rep (debounced) |
| label | enum | `correct` \| `incomplete_rom` \| `too_fast` \| `compensated` |
| dtw_similarity | 0–1 | similarity to the exercise reference rep |

Label assignment (proposed; to be confirmed in phase 3): `compensated` if any `TRUNK_*`,
`SHOULDER_*` or `ELBOW_*` flag was raised. Otherwise `too_fast` if TOO_FAST. Otherwise
`incomplete_rom` if INCOMPLETE_ROM. Otherwise `correct`. This mirrors the cue priority.

## 9. DTW similarity
1. Take the rep's smoothed primary-angle curve and z-normalize it (zero mean, unit variance).
2. Take the exercise reference rep curve from YAML (`reference.curve`) and z-normalize it the same way.
3. Compute DTW with a Sakoe–Chiba band (default 10 % of the longer length). Normalize the distance
   by the warping path length.
4. `similarity = 1 / (1 + normalized_distance)`.

Until real reference reps are recorded, the reference curves are **synthetic** (generated
minimum-jerk profiles), and every YAML says so (`reference.source: synthetic`).

## 10. Exercise YAML schema (sketch)
```yaml
id: shoulder_flexion
name: Shoulder flexion
view: sagittal              # sagittal | frontal
primary_angle: upper_arm_trunk_angle
target: 150                 # degrees
fsm: {start_below: 30, peak_above_offset: 15, return_below: 40}
min_rep_s: 2.0
max_peak_vel_dps: 90
rules:
  - {code: ELBOW_BEND, measure: elbow_interior, op: lt, value: 160, min_duration_s: 0.3, severity: warn}
  - {code: TRUNK_LEAN, measure: trunk_tilt, op: abs_delta_gt, value: 10, baseline: true, severity: warn}
  - {code: INCOMPLETE_ROM, scope: rep, op: peak_lt_frac_target, value: 0.9, severity: info}
  - {code: TOO_FAST, scope: rep, min_duration_s: 2.0, max_peak_vel_dps: 90, severity: warn}
  - {code: SHOULDER_HIKE, unknown_in_views: [sagittal]}
cue_texts: {ELBOW_BEND: "Keep your elbow straight", TRUNK_LEAN: "Keep your trunk upright"}
reference: {source: synthetic, fps: 30, curve: [...]}
```
The final schema is defined by the pydantic models in `rehabpose.core.rules` (phase 3).

## 11. Known limitations
- Monocular 2D angles depend on the view. Out-of-plane motion foreshortens angles.
- World `z` (depth) from a single camera is noisy, so depth-based rules are experimental.
- Loose clothing, low light, occlusion by furniture and wheelchairs degrade landmarks. The visibility
  gate mitigates this but cannot fully detect it.
- The defaults have not been validated on patients. See [09_ethics_privacy.md](09_ethics_privacy.md).
