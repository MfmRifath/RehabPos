import { Download, FileJson, Flame, Repeat, RotateCcw, Sparkles, Target } from "lucide-react";
import { useEffect, useState } from "react";
import { Skeleton, StatePanel, StatTile, Stepper } from "../components/ui";
import { api, type Report } from "../lib/api";
import { DISCLAIMER } from "../lib/disclaimer";
import { describeApiError } from "../lib/errors";
import { formatMeasure, plural, REP_LABELS, targetFraction } from "../lib/format";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";

export function Summary({ sid }: { sid: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useApp((s) => s.live);
  const { exercises } = useExercises();
  useEffect(() => {
    api.report(sid).then(setReport).catch((e: unknown) => setError(describeApiError(e)));
  }, [sid]);

  const fromThisSession = live.sid === sid;
  const stepper = fromThisSession && <Stepper current={3} />;

  if (error) {
    return (
      <section>
        <div className="page-head"><h1>Session summary</h1></div>
        <StatePanel tone="error" title="The report isn't available" action={<a className="btn btn-primary" href="#/">Back to exercises</a>}>
          {error}
        </StatePanel>
      </section>
    );
  }
  if (!report) {
    return (
      <section aria-busy="true">
        {stepper}
        <div className="page-head">
          <h1>Session summary</h1>
          <p className="muted" role="status">Loading report…</p>
        </div>
        <div className="stat-grid">
          <Skeleton className="skeleton-tile" />
          <Skeleton className="skeleton-tile" />
          <Skeleton className="skeleton-tile" />
        </div>
      </section>
    );
  }

  const s = report.summary;
  const { rom: target, unit } = report.target;
  const flagText = (code: string): string => exercises.find((e) => e.cue_texts[code])?.cue_texts[code] ?? code;
  const name = report.session?.exercise_name;

  return (
    <section>
      {stepper}
      <div className="page-head">
        <h1>Session summary</h1>
        <p className="lede">
          {s.rep_count > 0
            ? `${name ? `${name}: ` : ""}${plural(s.rep_count, "rep")} recorded.`
            : "No reps were counted this time. Check the camera setup and try again whenever you're ready."}
        </p>
      </div>
      <div className="stat-grid">
        <StatTile label="Reps" icon={<Repeat size={16} aria-hidden="true" />} value={s.rep_count} />
        <StatTile label="Mean peak" icon={<Target size={16} aria-hidden="true" />} value={formatMeasure(s.mean_rom, unit)} note={`target ${formatMeasure(target, unit)}`} />
        <StatTile label="Clean reps" icon={<Sparkles size={16} aria-hidden="true" />} value={s.quality_score === null ? "–" : `${Math.round(s.quality_score * 100)}%`} />
        {fromThisSession && (
          <StatTile label="Points" icon={<Flame size={16} aria-hidden="true" />} value={live.game.points} note={`best combo ${live.game.bestCombo}`} />
        )}
      </div>
      <div className="btn-row summary-actions">
        <a className="btn btn-primary" href="#/"><RotateCcw size={18} aria-hidden="true" />New session</a>
        <a className="btn" href={`/api/sessions/${sid}/report.pdf`} target="_blank" rel="noreferrer"><Download size={18} aria-hidden="true" />Download PDF report</a>
        <a className="btn btn-quiet" href={`/api/sessions/${sid}/report.json`} target="_blank" rel="noreferrer"><FileJson size={18} aria-hidden="true" />JSON</a>
      </div>
      {report.reps.length > 0 && (
        <>
          <h2 className="section-title">Repetitions</h2>
          <div className="card summary-table table-wrap" tabIndex={0} role="region" aria-label="Repetitions table">
            <table>
              <thead>
                <tr><th>#</th><th>Peak (target {formatMeasure(target, unit)})</th><th>Duration</th><th>Similarity</th><th>Result</th><th>Cues</th></tr>
              </thead>
              <tbody>
                {report.reps.map((r) => (
                  <tr key={r.idx}>
                    <td>{r.idx + 1}</td>
                    <td>
                      <div className="rep-bar">
                        <span className="rep-bar-value">{unit === "deg" ? `${r.peak.toFixed(1)}°` : r.peak.toFixed(2)}</span>
                        <div className="rep-bar-track" aria-hidden="true">
                          <div style={{ width: `${targetFraction(r.peak, target) * 100}%` }} />
                        </div>
                      </div>
                    </td>
                    <td>{r.duration_s.toFixed(1)} s</td>
                    <td>{r.similarity.toFixed(2)}</td>
                    <td><span className={`tag ${r.label === "correct" ? "tag-good" : ""}`}>{REP_LABELS[r.label]}</span></td>
                    <td className="wrap">{r.flags.map(flagText).join(", ") || "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small section-note">
            Similarity compares the shape of each rep with the exercise's reference movement, from 0 to 1. Higher is closer.
          </p>
        </>
      )}
      <p className="disclaimer small section-title">{DISCLAIMER}</p>
    </section>
  );
}
