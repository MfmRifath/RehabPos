// REST client for /api (same origin; proxied to FastAPI in dev).
import type { RepOut } from "../ws/protocol";

export interface RuleOut {
  code: string;
  kind: "frame" | "rep";
  measure?: string | null;
  op?: string | null;
  value?: number | null;
  severity: "info" | "warn" | "error";
  experimental: boolean;
}

export interface Exercise {
  id: string;
  name: string;
  view: "sagittal" | "frontal";
  allowed_views: ("sagittal" | "frontal")[];
  experimental: boolean;
  primary_angle: string;
  unit: string;
  target: number;
  min_rep_s: number;
  fsm: Record<string, number>;
  rules: RuleOut[];
  cue_texts: Record<string, string>;
}

export interface SessionOut {
  id: string;
  user_id: string;
  exercise_id: string;
  side: "L" | "R";
  view: string;
  source: string;
  started_at: string;
  ended_at: string | null;
  rep_count: number;
  mean_rom: number | null;
  quality_score: number | null;
  model_version: string;
}

export interface Target {
  exercise_id: string;
  target_rom?: number | null;
  min_rep_s?: number | null;
  thresholds: Record<string, number>;
}

export interface Job {
  id: string;
  status: "queued" | "running" | "done" | "error";
  progress: number;
  session_id: string;
  error: string | null;
  result: Record<string, string>;
}

export interface Badge {
  id: string;
  label: string;
  earned: boolean;
  detail: string;
}

export interface Progress {
  user_id: string;
  today: string;
  overall: { sessions: number; total_reps: number; active_days: number; current_streak: number; longest_streak: number };
  badges: Badge[];
  exercises: Record<
    string,
    {
      sessions: number;
      total_reps: number;
      best_mean_rom: number | null;
      rom_personal_best: { value: number; date: string } | null;
      series: { date: string; sessions: number; reps: number; mean_rom: number | null; quality: number | null }[];
    }
  >;
}

export interface Report {
  session?: { exercise_id: string; exercise_name: string };
  summary: { rep_count: number; mean_rom: number | null; best_rom: number | null; quality_score: number | null; labels: Record<string, number> };
  target: { rom: number; unit: string };
  reps: (Omit<RepOut, "peak_angle" | "t_peak"> & { peak: number })[];
  flags_summary: Record<string, number>;
  disclaimer: string;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = (await r.json()) as { detail?: unknown };
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, detail);
  }
  return (await r.json()) as T;
}

const json = (body: unknown, method = "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  exercises: () => req<Exercise[]>("/api/exercises"),
  createSession: (b: { user_id: string; exercise_id: string; side: "L" | "R"; view?: string }) =>
    req<SessionOut>("/api/sessions", json(b)),
  endSession: (sid: string) => req<SessionOut>(`/api/sessions/${sid}/end`, { method: "POST" }),
  report: (sid: string) => req<Report>(`/api/sessions/${sid}/report.json`),
  targets: (uid: string) => req<{ user_id: string; targets: Target[] }>(`/api/users/${uid}/targets`),
  putTargets: (uid: string, t: Target[]) => req<{ user_id: string; targets: Target[] }>(`/api/users/${uid}/targets`, json(t, "PUT")),
  /** Days are counted in the browser's time zone. */
  progress: (uid: string, tzOffsetMinutes = -new Date().getTimezoneOffset()) =>
    req<Progress>(`/api/users/${uid}/progress?tz_offset_minutes=${tzOffsetMinutes}`),
  exportUrl: (uid: string) => `/api/users/${uid}/export`,
  deleteUser: (uid: string) => req<{ user_id: string; deleted: Record<string, number> }>(`/api/users/${uid}`, { method: "DELETE" }),
  uploadVideo: (form: FormData) => req<{ job_id: string; session_id: string }>("/api/videos", { method: "POST", body: form }),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
};
