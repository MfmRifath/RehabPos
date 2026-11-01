"""Tiny synthetic files mimicking each dataset's ASSUMED layout (written into tmp dirs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rehabpose.data.joint_maps import KINECT_V2_JOINTS, PIG_MARKERS
from tests.fixtures.synthetic import (
    RepSpec,
    SynthPose,
    make_frame,
    rep_sequence,
    world_skeleton,
    write_frames_parquet,
)

F = NDArray[np.float64]

# MediaPipe index (or midpoint pair) used to synthesise each Kinect v2 joint
_KINECT_FROM_MP: dict[str, tuple[int, ...]] = {
    "SpineBase": (23, 24), "SpineMid": (11, 12, 23, 24), "Neck": (11, 12), "Head": (7, 8),
    "ShoulderLeft": (11,), "ElbowLeft": (13,), "WristLeft": (15,), "HandLeft": (19,),
    "ShoulderRight": (12,), "ElbowRight": (14,), "WristRight": (16,), "HandRight": (20,),
    "HipLeft": (23,), "KneeLeft": (25,), "AnkleLeft": (27,), "FootLeft": (31,),
    "HipRight": (24,), "KneeRight": (26,), "AnkleRight": (28,), "FootRight": (32,),
    "SpineShoulder": (11, 12), "HandTipLeft": (19,), "ThumbLeft": (21,),
    "HandTipRight": (20,), "ThumbRight": (22,),
}  # fmt: skip


def kinect_skeleton(poses: list[SynthPose], sensor_dist_m: float = 2.5) -> F:
    """(T, 25, 3) Kinect camera-space skeletons (metres, +y up, +x sensor-left, +z away)."""
    out = np.zeros((len(poses), 25, 3))
    for t, p in enumerate(poses):
        w = world_skeleton(p)  # MediaPipe world: y down, +x subject-left
        for j, name in enumerate(KINECT_V2_JOINTS):
            out[t, j] = w[list(_KINECT_FROM_MP[name])].mean(axis=0)
    out = out * np.array([-1.0, -1.0, 1.0])  # inverse of the loader's conversion
    out[..., 2] += sensor_dist_m
    out[..., 1] += 0.8  # sensor height offset (removed by hip centring)
    return out


def pig_markers(poses: list[SynthPose], d: float = 0.03) -> F:
    """(T, 18, 3) Plug-in Gait markers in a z-up lab frame (millimetres)."""
    out = np.zeros((len(poses), len(PIG_MARKERS), 3))
    fwd = np.array([0.0, 0.0, -d])
    for t, p in enumerate(poses):
        w = world_skeleton(p)
        m = {
            "LFHD": w[7] + fwd, "LBHD": w[7] - fwd, "RFHD": w[8] + fwd, "RBHD": w[8] - fwd,
            "LSHO": w[11], "RSHO": w[12], "LELB": w[13], "RELB": w[14],
            "LWRA": w[15] + fwd, "LWRB": w[15] - fwd, "RWRA": w[16] + fwd, "RWRB": w[16] - fwd,
            "LFIN": w[19], "RFIN": w[20],
            "LASI": w[23] + fwd, "LPSI": w[23] - fwd, "RASI": w[24] + fwd, "RPSI": w[24] - fwd,
        }  # fmt: skip
        for j, name in enumerate(PIG_MARKERS):
            x, y, z = m[name]
            out[t, j] = [x, z, -y]  # inverse of (x, -z, y)
    return out * 1000.0


def _arm_poses(n: int = 20, plane: float = 90.0) -> list[SynthPose]:
    return [SynthPose(elevation_deg=float(e), plane_deg=plane) for e in np.linspace(0, 120, n)]


def write_kinect_txt(path: Path, skel: F, with_state: bool = True, header: bool = True) -> Path:
    """Rows of 25 x (x y z [state]) numbers, optional header line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# frame data: x y z state per joint"] if header else []
    for frame in skel:
        vals = np.column_stack([frame, np.full(25, 2.0)]) if with_state else frame
        lines.append(" ".join(f"{v:.5f}" for v in vals.ravel()))
    path.write_text("\n".join(lines) + "\n")
    return path


def make_intellirehab(root: Path) -> Path:
    """Two subjects, two files each (IntelliRehabDS assumed naming)."""
    base = root / "raw" / "intellirehab"
    skel = kinect_skeleton(_arm_poses())
    for subj in ("101", "102"):
        for rep, ok in (("1", "1"), ("2", "2")):
            write_kinect_txt(base / f"{subj}_20200101_4_{rep}_{ok}_stand.txt", skel)
    (base / "README.txt").write_text("not a sequence")
    return base


def make_kimore(root: Path) -> Path:
    """KIMORE assumed tree with a comma-separated JointPosition CSV."""
    base = root / "raw" / "kimore"
    skel = kinect_skeleton(_arm_poses(10))
    f = base / "CG" / "Expert" / "E_ID1" / "Es1" / "Raw" / "JointPosition010.csv"
    f.parent.mkdir(parents=True)
    rows = [
        ",".join(f"{v:.5f}" for v in np.column_stack([fr, np.full(25, 2.0)]).ravel()) for fr in skel
    ]
    f.write_text("\n".join(rows) + "\n")
    return base


def make_uiprmd(root: Path) -> Path:
    """UI-PRMD assumed tree; 22 joints x 3 columns (order unknown to the loader)."""
    base = root / "raw" / "uiprmd"
    rng = np.random.default_rng(0)
    for split, suffix in (("Segmented Movements", ""), ("Incorrect Segmented Movements", "_inc")):
        f = base / split / "Kinect" / "Positions" / f"m07_s01_e01_positions{suffix}.txt"
        f.parent.mkdir(parents=True)
        f.write_text(
            "\n".join(",".join(f"{v:.4f}" for v in row) for row in rng.normal(size=(5, 66)))
        )
    return base


def make_trsp(root: Path) -> Path:
    """TRSP assumed CSV with 25x3 joint columns and a label column."""
    import pandas as pd

    base = root / "raw" / "trsp"
    skel = kinect_skeleton(_arm_poses(6))
    cols = {
        f"{n}_{a}": skel[:, j, k]
        for j, n in enumerate(KINECT_V2_JOINTS)
        for k, a in enumerate("xyz")
    }
    df = pd.DataFrame(cols)
    df["label"] = [
        "none",
        "none",
        "lean-forward",
        "lean-forward",
        "shoulder-elevation",
        "trunk-rotation",
    ]
    (base / "S01").mkdir(parents=True)
    df.to_csv(base / "S01" / "seq1.csv", index=False)
    return base


def make_videos(root: Path, name: str, rel_paths: list[str], n_frames: int = 5) -> Path:
    """Tiny stick-figure videos at the given root-relative paths."""
    from tests.fixtures.synthetic import write_frames_video

    base = root / "raw" / name
    frames = [
        make_frame(SynthPose(), "frontal", img_w=160, img_h=120, px_per_m=60, ts_ms=i * 33)
        for i in range(n_frames)
    ]
    for rel in rel_paths:
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        write_frames_video(base / rel, frames, style="stick")
    return base


def make_own(root: Path, n_subjects: int = 2) -> Path:
    """Own-data labels + landmark parquet for a short labelled set per subject."""
    from rehabpose.data.schema import Annotation, RepAnnotation, save_annotation

    for i in range(1, n_subjects + 1):
        vid = f"P{i:02d}_shoulder_flexion_R_sagittal_s1"
        frames = rep_sequence("shoulder_flexion", [RepSpec(150, duration_s=4.0)], seed=i)
        (root / "interim" / "own").mkdir(parents=True, exist_ok=True)
        write_frames_parquet(root / "interim" / "own" / f"{vid}.parquet", frames)
        ann = Annotation(
            video_id=vid, subject=f"P{i:02d}", exercise_id="shoulder_flexion", side="R",
            view="sagittal", fps=30.0, annotator="A1",
            reps=[RepAnnotation(
                idx=0, start_frame=50, peak_frame=105, end_frame=160, label="correct"
            )],
        )  # fmt: skip
        save_annotation(ann, root / "labels" / "own" / f"{vid}.json")
    return root
