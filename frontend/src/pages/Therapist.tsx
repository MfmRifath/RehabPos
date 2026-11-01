import { useEffect, useState } from "react";
import { api, type Target } from "../lib/api";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";

type Draft = Record<string, { target_rom: string; min_rep_s: string; thresholds: Record<string, string> }>;

const num = (s: string): number | null => (s.trim() === "" ? null : Number(s));

export function Therapist() {
  const userId = useApp((s) => s.profile.userId);
  const setUserId = useApp((s) => s.setUserId);
  const { exercises } = useExercises();
  const [draft, setDraft] = useState<Draft>({});
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    api
      .targets(userId)
      .then((t) => {
        const d: Draft = {};
        for (const x of t.targets) {
          d[x.exercise_id] = {
            target_rom: x.target_rom?.toString() ?? "",
            min_rep_s: x.min_rep_s?.toString() ?? "",
            thresholds: Object.fromEntries(Object.entries(x.thresholds).map(([k, v]) => [k, String(v)])),
          };
        }
        setDraft(d);
      })
      .catch(() => setDraft({}));
  }, [userId]);

  const edit = (eid: string, field: "target_rom" | "min_rep_s", v: string) =>
    setDraft((d) => ({ ...d, [eid]: { ...(d[eid] ?? { target_rom: "", min_rep_s: "", thresholds: {} }), [field]: v } }));
  const editThr = (eid: string, code: string, v: string) =>
    setDraft((d) => {
      const cur = d[eid] ?? { target_rom: "", min_rep_s: "", thresholds: {} };
      return { ...d, [eid]: { ...cur, thresholds: { ...cur.thresholds, [code]: v } } };
    });

  const save = async () => {
    const body: Target[] = Object.entries(draft).map(([eid, d]) => ({
      exercise_id: eid,
      target_rom: num(d.target_rom),
      min_rep_s: num(d.min_rep_s),
      thresholds: Object.fromEntries(Object.entries(d.thresholds).filter(([, v]) => v.trim() !== "").map(([k, v]) => [k, Number(v)])),
    }));
    try {
      await api.putTargets(userId, body);
      setMsg("Saved. New sessions use these settings.");
    } catch (e) {
      setMsg(`Not saved: ${String(e)}`);
    }
  };

  return (
    <section>
      <h1>Therapist settings</h1>
      <p className="muted">
        Per-user overrides of the engineering defaults. Leave a field empty to keep the default. These are
        not clinical prescriptions.
      </p>
      <label className="row">
        User id <input value={userId} onChange={(e) => setUserId(e.target.value)} pattern="[A-Za-z0-9_-]{1,64}" />
      </label>
      {exercises.map((ex) => (
        <div className="card" key={ex.id} style={{ marginTop: 12 }}>
          <h2 style={{ marginTop: 0 }}>{ex.name}</h2>
          <div className="row">
            <label>Target ROM <input type="number" placeholder={String(ex.target)} value={draft[ex.id]?.target_rom ?? ""} onChange={(e) => edit(ex.id, "target_rom", e.target.value)} /></label>
            <label>Min rep duration (s) <input type="number" step="0.1" placeholder={String(ex.min_rep_s)} value={draft[ex.id]?.min_rep_s ?? ""} onChange={(e) => edit(ex.id, "min_rep_s", e.target.value)} /></label>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            {ex.rules.filter((r) => r.kind === "frame").map((r) => (
              <label key={r.code}>
                {r.code}
                {r.experimental ? " (exp.)" : ""}{" "}
                <input type="number" step="0.01" placeholder={String(r.value)} value={draft[ex.id]?.thresholds[r.code] ?? ""} onChange={(e) => editThr(ex.id, r.code, e.target.value)} style={{ width: 90 }} />
              </label>
            ))}
          </div>
        </div>
      ))}
      <p className="row"><button className="primary" onClick={save}>Save</button>{msg && <span>{msg}</span>}</p>
    </section>
  );
}
