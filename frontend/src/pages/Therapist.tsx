import { ChevronDown, CircleAlert, Save } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { Skeleton, StatePanel } from "../components/ui";
import { api, ApiError, type Target } from "../lib/api";
import { describeApiError } from "../lib/errors";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";

type Draft = Record<string, { target_rom: string; min_rep_s: string; thresholds: Record<string, string> }>;

const num = (s: string): number | null => (s.trim() === "" ? null : Number(s));
const USER_ID = /^[A-Za-z0-9_-]{1,64}$/;

export function Therapist() {
  const userId = useApp((s) => s.profile.userId);
  const setUserId = useApp((s) => s.setUserId);
  const { exercises, error: exercisesError, loading, reload } = useExercises();
  const [userDraft, setUserDraft] = useState(userId);
  const [draft, setDraft] = useState<Draft>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .targets(userId)
      .then((t) => {
        if (cancelled) return;
        const d: Draft = {};
        for (const x of t.targets) {
          d[x.exercise_id] = {
            target_rom: x.target_rom?.toString() ?? "",
            min_rep_s: x.min_rep_s?.toString() ?? "",
            thresholds: Object.fromEntries(Object.entries(x.thresholds).map(([k, v]) => [k, String(v)])),
          };
        }
        setDraft(d);
        setDirty(false);
        setLoadError(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setDraft({});
        setDirty(false);
        // A new user simply has no overrides. Anything else: saving could overwrite settings we never saw.
        setLoadError(e instanceof ApiError && e.status === 404 ? null : describeApiError(e));
      });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  const edit = (eid: string, field: "target_rom" | "min_rep_s", v: string) => {
    setDirty(true);
    setMsg(null);
    setDraft((d) => ({ ...d, [eid]: { ...(d[eid] ?? { target_rom: "", min_rep_s: "", thresholds: {} }), [field]: v } }));
  };
  const editThr = (eid: string, code: string, v: string) => {
    setDirty(true);
    setMsg(null);
    setDraft((d) => {
      const cur = d[eid] ?? { target_rom: "", min_rep_s: "", thresholds: {} };
      return { ...d, [eid]: { ...cur, thresholds: { ...cur.thresholds, [code]: v } } };
    });
  };

  const save = async (e: FormEvent) => {
    e.preventDefault();
    const body: Target[] = Object.entries(draft).map(([eid, d]) => ({
      exercise_id: eid,
      target_rom: num(d.target_rom),
      min_rep_s: num(d.min_rep_s),
      thresholds: Object.fromEntries(Object.entries(d.thresholds).filter(([, v]) => v.trim() !== "").map(([k, v]) => [k, Number(v)])),
    }));
    setSaving(true);
    try {
      await api.putTargets(userId, body);
      setMsg("Saved. New sessions use these settings.");
      setDirty(false);
    } catch (err) {
      setMsg(`Not saved: ${describeApiError(err)}`);
    } finally {
      setSaving(false);
    }
  };

  const userValid = USER_ID.test(userDraft);
  const switchUser = (e: FormEvent) => {
    e.preventDefault();
    if (userValid) setUserId(userDraft);
  };

  return (
    <section>
      <div className="page-head">
        <h1>Therapist settings</h1>
        <p className="lede">
          Per-user overrides of the engineering defaults. Leave a field empty to keep the default. These are
          not clinical prescriptions.
        </p>
      </div>
      <div className="therapist-form">
        <form className="card user-card" onSubmit={switchUser}>
          <label className="field">
            User id
            <input className="input" value={userDraft} onChange={(e) => setUserDraft(e.target.value)} aria-invalid={!userValid} aria-describedby="user-id-help" />
            <small id="user-id-help" className={userValid ? "" : "field-error"}>
              {userValid ? "Letters, numbers, - and _ (up to 64 characters)." : "Use only letters, numbers, - and _ (1 to 64 characters)."}
            </small>
          </label>
          <button className="btn" type="submit" disabled={!userValid || userDraft === userId}>Switch user</button>
        </form>
        {loadError && (
          <p className="note note-critical" role="alert">
            <CircleAlert size={20} aria-hidden="true" />
            <span>Saved settings didn't load, so saving now could overwrite them. {loadError}</span>
          </p>
        )}
        {exercisesError ? (
          <StatePanel tone="error" title="Exercises didn't load" action={<button className="btn btn-primary" onClick={reload}>Try again</button>}>
            {exercisesError}
          </StatePanel>
        ) : loading ? (
          <div className="stack" aria-busy="true">
            <Skeleton className="skeleton-row" />
            <Skeleton className="skeleton-row" />
            <Skeleton className="skeleton-row" />
          </div>
        ) : (
          <form className="therapist-form" onSubmit={save}>
            {exercises.map((ex, i) => {
              const frameRules = ex.rules.filter((r) => r.kind === "frame");
              return (
                <details className="card disclosure" key={ex.id} open={i === 0}>
                  <summary>
                    {ex.name}
                    <ChevronDown size={18} aria-hidden="true" />
                  </summary>
                  <div className="rule-grid">
                    <label className="field">
                      Target range ({ex.unit === "deg" ? "degrees" : "ratio"})
                      <input className="input" type="number" step="any" placeholder={String(ex.target)} value={draft[ex.id]?.target_rom ?? ""} onChange={(e) => edit(ex.id, "target_rom", e.target.value)} />
                    </label>
                    <label className="field">
                      Shortest rep (seconds)
                      <input className="input" type="number" step="0.1" placeholder={String(ex.min_rep_s)} value={draft[ex.id]?.min_rep_s ?? ""} onChange={(e) => edit(ex.id, "min_rep_s", e.target.value)} />
                    </label>
                    {frameRules.map((r) => (
                      <label className="field" key={r.code}>
                        <span>
                          {ex.cue_texts[r.code] ?? r.code}
                          {r.experimental && <span className="tag tag-warn"> Experimental</span>}
                          <code>{r.code}</code>
                        </span>
                        <input className="input" type="number" step="0.01" placeholder={String(r.value ?? "")} value={draft[ex.id]?.thresholds[r.code] ?? ""} onChange={(e) => editThr(ex.id, r.code, e.target.value)} />
                      </label>
                    ))}
                  </div>
                </details>
              );
            })}
            <div className="save-bar">
              <span role="status">{msg ?? (dirty ? "You have unsaved changes." : "No unsaved changes.")}</span>
              <button className="btn btn-primary" type="submit" disabled={saving || !dirty}>
                <Save size={18} aria-hidden="true" />
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </form>
        )}
      </div>
    </section>
  );
}
