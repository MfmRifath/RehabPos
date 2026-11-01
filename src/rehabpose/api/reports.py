"""Session reports: structured JSON and a PDF (reportlab) with an ROM chart (matplotlib)."""

from __future__ import annotations

import io
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, col, select

from rehabpose import DISCLAIMER, __version__
from rehabpose.api.db import Event, Rep, SessionRow
from rehabpose.core.exercise import ExerciseConfig

SERIES = "#2a78d6"  # categorical slot 1 (validated on the light surface)
INK, INK_MUTED, GRID = "#0b0b0b", "#5f5e58", "#e4e3df"


def build_report(db: Session, row: SessionRow, exercise: ExerciseConfig) -> dict[str, Any]:
    """Structured report for one session (all numbers come from the stored session)."""
    reps = db.exec(select(Rep).where(Rep.session_id == row.id).order_by(col(Rep.idx))).all()
    events = db.exec(select(Event).where(Event.session_id == row.id).order_by(col(Event.t))).all()
    labels = Counter(r.label for r in reps)
    unit = "deg" if exercise.unit == "deg" else "ratio"
    return {
        "report_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "session": {
            "id": row.id,
            "user_id": row.user_id,
            "exercise_id": row.exercise_id,
            "exercise_name": exercise.name,
            "experimental": exercise.experimental,
            "side": row.side,
            "view": row.view,
            "source": row.source,
            "started_at": row.started_at.isoformat(timespec="seconds"),
            "ended_at": row.ended_at.isoformat(timespec="seconds") if row.ended_at else None,
            "model_version": row.model_version,
        },
        "target": {"rom": exercise.target, "unit": unit, "min_rep_s": exercise.min_rep_s},
        "summary": {
            "rep_count": len(reps),
            "mean_rom": round(sum(r.peak_angle for r in reps) / len(reps), 2) if reps else None,
            "best_rom": max((r.peak_angle for r in reps), default=None),
            "mean_similarity": (
                round(sum(r.similarity for r in reps) / len(reps), 4) if reps else None
            ),
            "quality_score": round(labels.get("correct", 0) / len(reps), 4) if reps else None,
            "labels": dict(labels),
        },
        "reps": [
            {
                "idx": r.idx,
                "t_start": r.t_start,
                "t_end": r.t_end,
                "peak": r.peak_angle,
                "duration_s": r.duration_s,
                "peak_vel": r.peak_vel,
                "similarity": r.similarity,
                "label": r.label,
                "flags": list(r.flags),
            }
            for r in reps
        ],
        "flags_summary": dict(Counter(f for r in reps for f in r.flags)),
        "events": [
            {"t": e.t, "code": e.code, "severity": e.severity, "source": e.source} for e in events
        ],
        "versions": {"rehabpose": __version__},
    }


def rom_chart_png(report: dict[str, Any], width_in: float = 6.5, height_in: float = 2.6) -> bytes:
    """Bar chart of peak ROM per rep with the target as a labelled reference line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = report["reps"]
    target = float(report["target"]["rom"])
    unit = "°" if report["target"]["unit"] == "deg" else ""
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=200)
    xs = [r["idx"] + 1 for r in reps]
    ys = [r["peak"] for r in reps]
    ax.bar(xs, ys, width=0.6, color=SERIES, edgecolor="white", linewidth=1.0, zorder=2)
    ax.axhline(target, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.annotate(
        f"Target {target:g}{unit}",
        xy=(1.0, target),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.set_title(
        f"Peak range of motion per rep ({unit or 'ratio'})", loc="left", fontsize=9, color=INK
    )
    ax.set_xlabel("Rep", fontsize=8, color=INK_MUTED)
    ax.set_xticks(xs if len(xs) <= 30 else xs[:: max(1, len(xs) // 15)])
    ax.set_ylim(0, max([target, *ys]) * 1.15 if ys or target else 1)
    ax.tick_params(colors=INK_MUTED, labelsize=7, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if not reps:
        ax.text(0.5, 0.5, "No completed reps", transform=ax.transAxes, ha="center", color=INK_MUTED)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def render_pdf(report: dict[str, Any]) -> bytes:
    """A4 PDF: header, disclaimer, summary, ROM chart, per-rep table, flags, version info."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    s, summ = report["session"], report["summary"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm,
        title=f"RehabPose session {s['id']}", author="RehabPose",
    )  # fmt: skip
    unit = "°" if report["target"]["unit"] == "deg" else ""
    story: list[Any] = [
        Paragraph(f"RehabPose session report: {s['exercise_name']}", styles["Title"]),
        Paragraph(
            f"Session {s['id']} · user {s['user_id']} · side {s['side']} · {s['view']} view · "
            f"{s['source']} · started {s['started_at']}",
            styles["Normal"],
        ),
        Spacer(1, 0.3 * cm),
    ]
    disclaimer = Table(
        [[Paragraph(f"<b>Disclaimer.</b> {report['disclaimer']}", styles["Normal"])]]
    )
    disclaimer.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#b4532a")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf3ec")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )  # fmt: skip
    )
    story += [disclaimer, Spacer(1, 0.4 * cm)]

    def fmt(v: Any, nd: int = 1) -> str:
        return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    mean_peak = f"{fmt(summ['mean_rom'])}{unit}"
    best_peak = f"{fmt(summ['best_rom'])}{unit}"
    summary_rows = [
        ["Reps", fmt(summ["rep_count"]), "Target", f"{report['target']['rom']:g}{unit}"],
        ["Mean peak", mean_peak, "Best peak", best_peak],
        [
            "Quality (correct reps)",
            fmt(summ["quality_score"], 2),
            "Mean similarity",
            fmt(summ["mean_similarity"], 3),
        ],
    ]
    t = Table(summary_rows, colWidths=[4.2 * cm, 3.6 * cm, 4.2 * cm, 3.6 * cm])
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                ("TEXTCOLOR", (2, 0), (2, -1), colors.grey),
            ]
        )
    )
    story += [t, Spacer(1, 0.3 * cm)]
    story.append(Image(io.BytesIO(rom_chart_png(report)), width=17 * cm, height=6.8 * cm))
    story.append(Spacer(1, 0.3 * cm))

    header = ["#", f"Peak{unit}", "Duration s", "Peak vel /s", "Similarity", "Label", "Flags"]
    rows = [header] + [
        [
            str(r["idx"] + 1), f"{r['peak']:.1f}", f"{r['duration_s']:.2f}", f"{r['peak_vel']:.0f}",
            f"{r['similarity']:.3f}", r["label"], ", ".join(r["flags"]) or "–",
        ]
        for r in report["reps"]
    ]  # fmt: skip
    reps_table = Table(rows, repeatRows=1)
    reps_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f2")]),
            ]
        )  # fmt: skip
    )
    story += [Paragraph("Repetitions", styles["Heading3"]), reps_table if report["reps"] else
              Paragraph("No completed reps.", styles["Normal"])]  # fmt: skip

    flags = report["flags_summary"]
    flag_text = ", ".join(f"{k}: {v}" for k, v in sorted(flags.items())) or "none"
    story += [
        Paragraph("Flags", styles["Heading3"]),
        Paragraph(
            f"Reps with each flag: {flag_text}. Flag events: {len(report['events'])}.",
            styles["Normal"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph(
            f"Model/version: {s['model_version'] or 'n/a'} · report v{report['report_version']} · "
            f"generated {report['generated_at']}",
            styles["Italic"],
        ),
    ]
    doc.build(story)
    return buf.getvalue()
