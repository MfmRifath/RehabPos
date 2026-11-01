import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Progress } from "../lib/api";
import { newUserId, useApp } from "../store/app";
import { useExercises } from "../useExercises";

const axis = { stroke: "var(--ink-muted)", fontSize: 12 };
const grid = <CartesianGrid stroke="var(--line)" vertical={false} />;
const tip = {
  contentStyle: { background: "var(--surface)", border: "1px solid var(--line)", color: "var(--ink)" },
  cursor: { stroke: "var(--ink-muted)", fill: "var(--surface-2)" },
};

function YourData({ userId, onDeleted }: { userId: string; onDeleted: () => void }) {
  const [message, setMessage] = useState<string | null>(null);
  const remove = async () => {
    if (!window.confirm(`Permanently delete all data for ${userId}? This cannot be undone.`)) return;
    try {
      const r = await api.deleteUser(userId);
      setMessage(`Deleted ${r.deleted.sessions ?? 0} sessions and ${r.deleted.reps ?? 0} reps.`);
      onDeleted();
    } catch (e) {
      setMessage(`Not deleted: ${String(e)}`);
    }
  };
  return (
    <div className="card" style={{ marginTop: 16 }} data-testid="your-data">
      <h2 style={{ marginTop: 0 }}>Your data</h2>
      <p className="muted">Everything is stored on this RehabPose server under a pseudonymous id ({userId}).</p>
      <div className="row">
        <a href={api.exportUrl(userId)} download>Export as JSON</a>
        <button onClick={remove} data-testid="delete-data">Delete all my data</button>
        {message && <span role="status">{message}</span>}
      </div>
    </div>
  );
}

export function ProgressPage() {
  const userId = useApp((s) => s.profile.userId);
  const setUserId = useApp((s) => s.setUserId);
  const { exercises } = useExercises();
  const [data, setData] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api
      .progress(userId)
      .then(setData)
      .catch((e: unknown) => setError(String(e).includes("404") || (e as { status?: number }).status === 404 ? "No sessions yet." : String(e)));
  }, [userId]);

  if (error) return <section><h1>Progress</h1><p className="muted">{error}</p></section>;
  if (!data) return <p className="muted">Loading…</p>;
  const yours = (
    <YourData
      userId={userId}
      onDeleted={() => {
        setData(null);
        setUserId(newUserId());
      }}
    />
  );
  const entries = Object.entries(data.exercises);
  if (entries.length === 0) return <section><h1>Progress</h1><p className="muted">No completed sessions yet.</p>{yours}</section>;
  const o = data.overall;

  return (
    <section>
      <h1>Progress</h1>
      <p data-testid="overall">
        {o.total_reps} reps in {o.sessions} sessions · streak {o.current_streak} days (longest {o.longest_streak})
      </p>
      <div className="row" aria-label="Badges" style={{ marginBottom: 16 }}>
        {data.badges.map((b) => (
          <span key={b.id} className={`badge ${b.earned ? "earned" : ""}`} title={b.detail}>
            {b.earned ? "🏅" : "○"} {b.label}
          </span>
        ))}
      </div>
      {entries.map(([eid, p]) => {
        const ex = exercises.find((e) => e.id === eid);
        const unit = ex?.unit === "deg" ? "°" : "";
        const series = p.series.map((d) => ({ ...d, qualityPct: d.quality === null ? null : Math.round(d.quality * 100) }));
        return (
          <div className="card" key={eid} style={{ marginBottom: 16 }}>
            <h2 style={{ marginTop: 0 }}>{ex?.name ?? eid}</h2>
            <p className="muted">
              {p.sessions} sessions · {p.total_reps} reps
              {p.rom_personal_best && ` · best session ${p.rom_personal_best.value}${unit} on ${p.rom_personal_best.date}`}
            </p>
            <h3>Mean peak ROM per day ({unit || "ratio"})</h3>
            <div className="chart">
              <ResponsiveContainer>
                <LineChart data={series} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
                  {grid}
                  <XAxis dataKey="date" tick={axis} />
                  <YAxis tick={axis} domain={[0, "auto"]} />
                  <Tooltip {...tip} />
                  {ex && <ReferenceLine y={ex.target} stroke="var(--ink-muted)" strokeDasharray="4 3" label={{ value: `target ${ex.target}${unit}`, fill: "var(--ink-muted)", fontSize: 12, position: "insideTopRight" }} />}
                  <Line type="monotone" dataKey="mean_rom" name="mean ROM" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4 }} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <h3>Reps per day</h3>
            <div className="chart">
              <ResponsiveContainer>
                <BarChart data={series} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
                  {grid}
                  <XAxis dataKey="date" tick={axis} />
                  <YAxis tick={axis} allowDecimals={false} />
                  <Tooltip {...tip} />
                  <Bar dataKey="reps" name="reps" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={36} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <h3>Clean reps per day (%)</h3>
            <div className="chart">
              <ResponsiveContainer>
                <LineChart data={series} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
                  {grid}
                  <XAxis dataKey="date" tick={axis} />
                  <YAxis tick={axis} domain={[0, 100]} />
                  <Tooltip {...tip} />
                  <Line type="monotone" dataKey="qualityPct" name="clean reps %" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4 }} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <details>
              <summary>Table view</summary>
              <table>
                <thead><tr><th>Date</th><th>Sessions</th><th>Reps</th><th>Mean ROM</th><th>Clean %</th></tr></thead>
                <tbody>
                  {series.map((d) => (
                    <tr key={d.date}><td>{d.date}</td><td>{d.sessions}</td><td>{d.reps}</td><td>{d.mean_rom ?? "–"}</td><td>{d.qualityPct ?? "–"}</td></tr>
                  ))}
                </tbody>
              </table>
            </details>
          </div>
        );
      })}
      {yours}
    </section>
  );
}
