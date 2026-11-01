import { useEffect, useState } from "react";
import { api, type Report } from "../lib/api";
import { DISCLAIMER } from "../lib/disclaimer";
import { useApp } from "../store/app";

export function Summary({ sid }: { sid: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useApp((s) => s.live);
  useEffect(() => {
    api.report(sid).then(setReport).catch((e: unknown) => setError(String(e)));
  }, [sid]);
  if (error) return <p className="error">{error}</p>;
  if (!report) return <p className="muted">Loading report…</p>;
  const s = report.summary;
  const unit = report.target.unit === "deg" ? "°" : "";
  return (
    <section>
      <h1>Session summary</h1>
      <div className="grid">
        <div className="card"><div className="muted">Reps</div><div style={{ fontSize: 32 }}>{s.rep_count}</div></div>
        <div className="card"><div className="muted">Mean peak</div><div style={{ fontSize: 32 }}>{s.mean_rom ?? "–"}{unit}</div><div className="muted">target {report.target.rom}{unit}</div></div>
        <div className="card"><div className="muted">Clean reps</div><div style={{ fontSize: 32 }}>{s.quality_score === null ? "–" : `${Math.round(s.quality_score * 100)}%`}</div></div>
        {live.sid === sid && <div className="card"><div className="muted">Points</div><div style={{ fontSize: 32 }}>{live.game.points}</div><div className="muted">best combo {live.game.bestCombo}</div></div>}
      </div>
      <h2>Repetitions</h2>
      <table>
        <thead><tr><th>#</th><th>Peak</th><th>Duration</th><th>Similarity</th><th>Label</th><th>Flags</th></tr></thead>
        <tbody>
          {report.reps.map((r) => (
            <tr key={r.idx}>
              <td>{r.idx + 1}</td><td>{r.peak.toFixed(1)}{unit}</td><td>{r.duration_s.toFixed(1)} s</td>
              <td>{r.similarity.toFixed(2)}</td><td>{r.label.replace("_", " ")}</td><td>{r.flags.join(", ") || "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="row" style={{ marginTop: 16 }}>
        <a href={`/api/sessions/${sid}/report.pdf`} target="_blank" rel="noreferrer">Download PDF report</a>
        <a href={`/api/sessions/${sid}/report.json`} target="_blank" rel="noreferrer">JSON</a>
        <a href="#/">New session</a>
      </p>
      <p className="muted disclaimer">{DISCLAIMER}</p>
    </section>
  );
}
