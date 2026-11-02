import { CalendarDays, ChevronDown, Download, Flame, Lock, Medal, Repeat, Trash2, Trophy } from "lucide-react";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Modal } from "../components/Modal";
import { Segmented, StatePanel, StatTile } from "../components/ui";
import { api, ApiError, type Exercise, type Progress } from "../lib/api";
import { describeApiError } from "../lib/errors";
import { plural, shortDate } from "../lib/format";
import { newUserId, useApp } from "../store/app";
import { useExercises } from "../useExercises";
import { ProgressLoading } from "./ProgressLoading";

const axis = { fill: "var(--ink-muted)", fontSize: 12 };
const axisLine = { stroke: "var(--line)" };
const grid = <CartesianGrid stroke="var(--line)" vertical={false} />;
const dot = { r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 };
const tip = {
  contentStyle: { background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 10, color: "var(--ink)" },
  labelFormatter: (d: unknown) => shortDate(String(d)),
  cursor: { stroke: "var(--line-strong)", fill: "var(--surface-2)" },
};
const margin = { top: 12, right: 12, bottom: 0, left: -8 };

type Metric = "rom" | "reps" | "clean";
type Row = Progress["exercises"][string]["series"][number] & { qualityPct: number | null };

const METRICS: { value: Metric; label: string }[] = [
  { value: "rom", label: "Range" },
  { value: "reps", label: "Reps" },
  { value: "clean", label: "Clean %" },
];

function MetricChart({ metric, series, ex, unit }: { metric: Metric; series: Row[]; ex?: Exercise; unit: string }) {
  const x = <XAxis dataKey="date" tick={axis} axisLine={axisLine} tickLine={false} tickFormatter={shortDate} minTickGap={24} />;
  if (metric === "reps") {
    return (
      <ResponsiveContainer>
        <BarChart data={series} margin={margin}>
          {grid}
          {x}
          <YAxis tick={axis} axisLine={false} tickLine={false} allowDecimals={false} />
          <Tooltip {...tip} />
          <Bar dataKey="reps" name="reps" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    );
  }
  const rom = metric === "rom";
  return (
    <ResponsiveContainer>
      <LineChart data={series} margin={margin}>
        {grid}
        {x}
        <YAxis tick={axis} axisLine={false} tickLine={false} domain={rom ? [0, "auto"] : [0, 100]} />
        <Tooltip {...tip} />
        {/* extendDomain: the target must stay on the chart while someone is still working up to it. */}
        {rom && ex && (
          <ReferenceLine y={ex.target} ifOverflow="extendDomain" stroke="var(--ink-muted)" strokeDasharray="4 3" label={{ value: `target ${ex.target}${unit}`, fill: "var(--ink-muted)", fontSize: 12, position: "insideBottomLeft" }} />
        )}
        <Line type="monotone" dataKey={rom ? "mean_rom" : "qualityPct"} name={rom ? "mean ROM" : "clean reps %"} stroke="var(--series-1)" strokeWidth={2} dot={dot} activeDot={{ ...dot, r: 6 }} connectNulls isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ExerciseCard({ id, p, ex }: { id: string; p: Progress["exercises"][string]; ex?: Exercise }) {
  const [metric, setMetric] = useState<Metric>("rom");
  const unit = ex?.unit === "deg" ? "°" : "";
  const series: Row[] = p.series.map((d) => ({ ...d, qualityPct: d.quality === null ? null : Math.round(d.quality * 100) }));
  const caption = { rom: `Mean peak range per day (${unit || "ratio"})`, reps: "Reps per day", clean: "Clean reps per day (%)" }[metric];
  return (
    <div className="card exercise-card">
      <div className="exercise-card-head">
        <div>
          <h2>{ex?.name ?? id}</h2>
          <p className="muted small">
            {plural(p.sessions, "session")} · {plural(p.total_reps, "rep")}
            {p.rom_personal_best && ` · best session ${p.rom_personal_best.value}${unit} on ${p.rom_personal_best.date}`}
          </p>
        </div>
        <Segmented legend="Show" name={`metric-${id}`} value={metric} options={METRICS} onChange={setMetric} />
      </div>
      <figure className="chart" aria-label={`${caption}. The same values are in the table below.`}>
        <figcaption className="chart-caption">{caption}</figcaption>
        <div className="chart-plot">
          <MetricChart metric={metric} series={series} ex={ex} unit={unit} />
        </div>
      </figure>
      <details className="disclosure">
        <summary>
          Table view
          <ChevronDown size={18} aria-hidden="true" />
        </summary>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Date</th><th>Sessions</th><th>Reps</th><th>Mean ROM</th><th>Clean %</th></tr></thead>
            <tbody>
              {series.map((d) => (
                <tr key={d.date}><td>{d.date}</td><td>{d.sessions}</td><td>{d.reps}</td><td>{d.mean_rom ?? "–"}</td><td>{d.qualityPct ?? "–"}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function YourData({ userId, hasData, notice, onDelete }: { userId: string; hasData: boolean; notice: string | null; onDelete: () => void }) {
  return (
    <div className="card your-data" data-testid="your-data">
      <h2>Your data</h2>
      <p className="muted">Everything is stored on this RehabPose server under a pseudonymous id (<code>{userId}</code>).</p>
      {hasData && (
        <div className="btn-row">
          <a className="btn" href={api.exportUrl(userId)} download><Download size={18} aria-hidden="true" />Export as JSON</a>
          <button className="btn btn-danger" onClick={onDelete} data-testid="delete-data"><Trash2 size={18} aria-hidden="true" />Delete all my data</button>
        </div>
      )}
      {notice && <p role="status">{notice}</p>}
    </div>
  );
}

export function ProgressPage() {
  const userId = useApp((s) => s.profile.userId);
  const setUserId = useApp((s) => s.setUserId);
  const { exercises } = useExercises();
  const [data, setData] = useState<Progress | null>(null);
  const [missing, setMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [confirming, setConfirming] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .progress(userId)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) setMissing(true);
        else setError(describeApiError(e));
      });
    return () => {
      cancelled = true;
    };
  }, [userId, attempt]);

  const remove = async () => {
    setConfirming(false);
    try {
      const r = await api.deleteUser(userId);
      setNotice(`Deleted ${r.deleted.sessions ?? 0} sessions and ${r.deleted.reps ?? 0} reps.`);
      setData(null);
      setUserId(newUserId());
    } catch (e) {
      setNotice(`Not deleted: ${describeApiError(e)}`);
    }
  };

  const noSessions = (
    <StatePanel tone="empty" title="No sessions yet" action={<a className="btn btn-primary" href="#/">Choose an exercise</a>}>
      Finish a session and your range, reps and streaks will show up here.
    </StatePanel>
  );

  if (error) {
    return (
      <section>
        <div className="page-head"><h1>Progress</h1></div>
        <StatePanel
          tone="error"
          title="Progress didn't load"
          action={
            <button
              className="btn btn-primary"
              onClick={() => {
                setError(null);
                setAttempt((a) => a + 1);
              }}
            >
              Try again
            </button>
          }
        >
          {error}
        </StatePanel>
      </section>
    );
  }
  if (missing) {
    return (
      <section>
        <div className="page-head"><h1>Progress</h1></div>
        {noSessions}
        <YourData userId={userId} hasData={false} notice={notice} onDelete={() => undefined} />
      </section>
    );
  }
  if (!data) return <ProgressLoading />;

  const entries = Object.entries(data.exercises);
  const o = data.overall;
  const yours = <YourData userId={userId} hasData notice={notice} onDelete={() => setConfirming(true)} />;
  const dialog = confirming && (
    <Modal labelledBy="delete-title" role="alertdialog" onDismiss={() => setConfirming(false)}>
      <h2 id="delete-title">Delete all your data?</h2>
      <p>This permanently deletes every session and report for <code>{userId}</code>. It cannot be undone.</p>
      <div className="btn-row">
        <button className="btn" onClick={() => setConfirming(false)} autoFocus data-testid="delete-cancel">Keep my data</button>
        <button className="btn btn-danger-solid" onClick={remove} data-testid="delete-confirm">Delete everything</button>
      </div>
    </Modal>
  );

  if (entries.length === 0) {
    return (
      <section>
        <div className="page-head"><h1>Progress</h1></div>
        {noSessions}
        {yours}
        {dialog}
      </section>
    );
  }

  return (
    <section>
      <div className="page-head">
        <h1>Progress</h1>
        <p className="lede" data-testid="overall">{o.total_reps} reps in {plural(o.sessions, "session")}</p>
      </div>
      <div className="stat-grid">
        <StatTile label="Total reps" icon={<Repeat size={16} aria-hidden="true" />} value={o.total_reps} />
        <StatTile label="Active days" icon={<CalendarDays size={16} aria-hidden="true" />} value={o.active_days} />
        <StatTile label="Current streak" icon={<Flame size={16} aria-hidden="true" />} value={plural(o.current_streak, "day")} />
        <StatTile label="Longest streak" icon={<Trophy size={16} aria-hidden="true" />} value={plural(o.longest_streak, "day")} />
      </div>
      <h2 className="section-title">Badges</h2>
      <ul className="badge-grid" aria-label="Badges">
        {data.badges.map((b) => (
          <li key={b.id} className={`badge ${b.earned ? "earned" : ""}`}>
            <span className="badge-icon">{b.earned ? <Medal size={20} aria-hidden="true" /> : <Lock size={18} aria-hidden="true" />}</span>
            <span>
              <span className="badge-label">{b.label}</span>
              <span className="badge-detail">{b.earned ? "Earned" : "Locked"} · {b.detail}</span>
            </span>
          </li>
        ))}
      </ul>
      <h2 className="section-title">By exercise</h2>
      <div className="stack">
        {entries.map(([eid, p]) => (
          <ExerciseCard key={eid} id={eid} p={p} ex={exercises.find((e) => e.id === eid)} />
        ))}
      </div>
      {yours}
      {dialog}
    </section>
  );
}
