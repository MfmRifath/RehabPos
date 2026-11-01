import { navigate } from "../router";
import { useApp, type ModelVariant } from "../store/app";
import { useExercises } from "../useExercises";

export function Home() {
  const { exercises, error } = useExercises();
  const sel = useApp((s) => s.selection);
  const select = useApp((s) => s.select);
  const ex = exercises.find((e) => e.id === sel.exerciseId);
  const views = ex?.allowed_views ?? ["sagittal", "frontal"];

  return (
    <section>
      <h1>Choose an exercise</h1>
      {error && <p className="error">{error}</p>}
      <div className="grid" role="list">
        {exercises.map((e) => (
          <button
            key={e.id}
            role="listitem"
            className="card exercise"
            aria-pressed={e.id === sel.exerciseId}
            data-testid={`exercise-${e.id}`}
            onClick={() => select({ exerciseId: e.id, view: null })}
          >
            <strong>{e.name}</strong>
            <span className="muted">
              {e.view} view · target {e.target}
              {e.unit === "deg" ? "°" : ""}
            </span>
            {e.experimental && <span className="tag">experimental</span>}
          </button>
        ))}
      </div>
      <div className="card row" style={{ marginTop: 16 }}>
        <label>
          Side{" "}
          <select value={sel.side} onChange={(e) => select({ side: e.target.value as "L" | "R" })} data-testid="side">
            <option value="R">Right arm</option>
            <option value="L">Left arm</option>
          </select>
        </label>
        <label>
          Camera view{" "}
          <select value={sel.view ?? ex?.view ?? ""} onChange={(e) => select({ view: e.target.value as "sagittal" | "frontal" })}>
            {views.map((v) => (
              <option key={v} value={v}>
                {v === "sagittal" ? "Side-on (sagittal)" : "Facing camera (frontal)"}
              </option>
            ))}
          </select>
        </label>
        <label>
          Pose model{" "}
          <select value={sel.model} onChange={(e) => select({ model: e.target.value as ModelVariant })}>
            <option value="lite">lite (fastest)</option>
            <option value="full">full</option>
            <option value="heavy">heavy (most accurate)</option>
          </select>
        </label>
        <button className="primary" disabled={!ex} onClick={() => navigate("/setup")} data-testid="go-setup">
          Set up camera
        </button>
      </div>
      <p className="muted">
        Earn points for clean reps and build combos. Small, immediate rewards are a common idea in gamified
        rehabilitation to keep home practice going; they are motivation only, not a measure of recovery.
      </p>
    </section>
  );
}
