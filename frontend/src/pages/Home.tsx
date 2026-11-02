import { ArrowRight, Check, ChevronDown } from "lucide-react";
import { ExercisePictogram } from "../components/ExercisePictogram";
import { Segmented, Skeleton, StatePanel, Stepper } from "../components/ui";
import { formatMeasure, viewLabel } from "../lib/format";
import { navigate } from "../router";
import { useApp, type ModelVariant, type Side } from "../store/app";
import { useExercises } from "../useExercises";

const SIDES = [
  { value: "R" as Side, label: "Right arm" },
  { value: "L" as Side, label: "Left arm" },
];

const MODELS = [
  { value: "lite" as ModelVariant, label: "Lite (fastest)" },
  { value: "full" as ModelVariant, label: "Full" },
  { value: "heavy" as ModelVariant, label: "Heavy (most accurate)" },
];

export function Home() {
  const { exercises, error, loading, reload } = useExercises();
  const sel = useApp((s) => s.selection);
  const select = useApp((s) => s.select);
  const ex = exercises.find((e) => e.id === sel.exerciseId);
  const views = ex?.allowed_views ?? [];

  return (
    <section>
      <Stepper current={0} />
      <div className="page-head">
        <h1>What are we working on today?</h1>
        <p className="lede">Pick an exercise, then check your camera. You'll get one short cue at a time while you move.</p>
      </div>
      {error ? (
        <StatePanel tone="error" title="Exercises didn't load" action={<button className="btn btn-primary" onClick={reload}>Try again</button>}>
          {error}
        </StatePanel>
      ) : !loading && exercises.length === 0 ? (
        <StatePanel tone="empty" title="No exercises available">
          The server has no exercises configured.
        </StatePanel>
      ) : (
        <div className="home-layout">
          {loading ? (
            <div className="exercise-grid" aria-busy="true">
              <Skeleton className="skeleton-card" />
              <Skeleton className="skeleton-card" />
              <Skeleton className="skeleton-card" />
            </div>
          ) : (
            <ul className="exercise-grid" aria-label="Exercises">
              {exercises.map((e) => {
                const selected = e.id === sel.exerciseId;
                return (
                  <li key={e.id}>
                    <button
                      className="exercise"
                      aria-pressed={selected}
                      data-testid={`exercise-${e.id}`}
                      onClick={() => select({ exerciseId: e.id, view: null })}
                    >
                      <ExercisePictogram id={e.id} />
                      <span className="exercise-name">{e.name}</span>
                      <span className="exercise-meta">
                        {viewLabel(e.view)} · target {formatMeasure(e.target, e.unit)}
                      </span>
                      {e.experimental && <span className="tag tag-warn">Experimental</span>}
                      {selected && (
                        <span className="exercise-tick" aria-hidden="true">
                          <Check size={18} strokeWidth={3} />
                        </span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          <aside className="card session-panel" aria-label="Session options">
            <h2>{ex?.name ?? "Choose an exercise"}</h2>
            <Segmented legend="Which arm?" name="side" value={sel.side} options={SIDES} onChange={(side) => select({ side })} testId="side" />
            {views.length > 1 && (
              <Segmented
                legend="Camera position"
                name="view"
                value={sel.view ?? ex?.view ?? views[0] ?? "sagittal"}
                options={views.map((v) => ({ value: v, label: viewLabel(v) }))}
                onChange={(view) => select({ view })}
              />
            )}
            <details className="disclosure">
              <summary>
                Advanced
                <ChevronDown size={18} aria-hidden="true" />
              </summary>
              <label className="field">
                Pose model
                <select className="select" value={sel.model} onChange={(e) => select({ model: e.target.value as ModelVariant })}>
                  {MODELS.map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
              </label>
            </details>
            <button className="btn btn-primary btn-lg btn-block" disabled={!ex} onClick={() => navigate("/setup")} data-testid="go-setup">
              Set up camera
              <ArrowRight size={20} aria-hidden="true" />
            </button>
            <p className="muted small">
              Earn points for clean reps and build combos. They are motivation only, not a measure of recovery.
            </p>
          </aside>
        </div>
      )}
    </section>
  );
}
