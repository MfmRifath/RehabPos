// WebSocket protocol v1 — mirrors src/rehabpose/api/schemas.py (keep in sync).

export const PROTOCOL_VERSION = 1 as const;

export type Phase = "rest" | "ascending" | "peak" | "descending";
export type Severity = "info" | "warn" | "error";
export type ControlAction = "start_set" | "pause" | "end_set" | "calibrate";
/** [x, y, z, visibility] */
export type Landmark4 = [number, number, number, number];

export interface LandmarksMsg {
  v: 1;
  type: "landmarks";
  sid: string;
  frame: number;
  ts_ms: number;
  img: { w: number; h: number };
  lm: Landmark4[];
  world: Landmark4[];
  client_infer_ms?: number;
}

export interface ControlMsg {
  v: 1;
  type: "control";
  action: ControlAction;
}

export interface FlagOut {
  code: string;
  severity: Severity;
  confidence: number;
  source: "rule" | "model";
}

export interface CueOut {
  code: string;
  text: string;
  speak: boolean;
  severity?: Severity | null;
}

export interface FeedbackMsg {
  v: 1;
  type: "feedback";
  frame: number;
  ts_ms: number;
  angles: Record<string, number | null>;
  phase: Phase;
  rep: { count: number; last_peak: number | null; last_duration_s: number | null; last_similarity: number | null };
  flags: FlagOut[];
  cue: CueOut | null;
  quality: { visibility_ok: boolean; view_ok: boolean };
  server_ms: number;
  dropped?: number;
}

export interface RepOut {
  idx: number;
  t_start: number;
  t_peak: number;
  t_end: number;
  peak_angle: number;
  duration_s: number;
  peak_vel: number;
  similarity: number;
  label: "correct" | "incomplete_rom" | "too_fast" | "compensated";
  flags: string[];
}

export interface RepCompletedMsg {
  v: 1;
  type: "rep_completed";
  rep: RepOut;
}

export interface SetupStatusMsg {
  v: 1;
  type: "setup_status";
  calibrated: boolean;
  visibility_ok: boolean;
  view_ok: boolean;
  hints: string[];
}

export interface ErrorMsg {
  v: 1;
  type: "error";
  code: string;
  message: string;
}

export type ServerMessage = FeedbackMsg | RepCompletedMsg | SetupStatusMsg | ErrorMsg;

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null;

/** Parse and minimally validate a server message; returns null for anything malformed. */
export function parseServerMessage(raw: string): ServerMessage | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isObj(data) || data.v !== PROTOCOL_VERSION || typeof data.type !== "string") return null;
  switch (data.type) {
    case "feedback":
      return typeof data.frame === "number" && isObj(data.rep) && isObj(data.quality) && Array.isArray(data.flags)
        ? (data as unknown as FeedbackMsg)
        : null;
    case "rep_completed":
      return isObj(data.rep) && typeof (data.rep as { idx?: unknown }).idx === "number"
        ? (data as unknown as RepCompletedMsg)
        : null;
    case "setup_status":
      return typeof data.calibrated === "boolean" && Array.isArray(data.hints)
        ? (data as unknown as SetupStatusMsg)
        : null;
    case "error":
      return typeof data.code === "string" ? (data as unknown as ErrorMsg) : null;
    default:
      return null;
  }
}

interface LandmarkLike {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

const toRow = (p: LandmarkLike): Landmark4 => [p.x, p.y, p.z, p.visibility ?? 0];

/** Build a landmarks message from MediaPipe output (33 image + 33 world landmarks). */
export function landmarksMessage(args: {
  sid: string;
  frame: number;
  tsMs: number;
  width: number;
  height: number;
  landmarks: LandmarkLike[];
  world: LandmarkLike[];
  inferMs?: number;
}): LandmarksMsg {
  if (args.landmarks.length !== 33 || args.world.length !== 33) {
    throw new Error(`expected 33 landmarks, got ${args.landmarks.length}/${args.world.length}`);
  }
  const world = args.world.map(toRow);
  // World landmarks carry the image landmarks' visibility (as the Python backend does).
  args.landmarks.forEach((p, i) => {
    (world[i] as Landmark4)[3] = p.visibility ?? 0;
  });
  return {
    v: 1,
    type: "landmarks",
    sid: args.sid,
    frame: args.frame,
    ts_ms: Math.round(args.tsMs),
    img: { w: args.width, h: args.height },
    lm: args.landmarks.map(toRow),
    world,
    client_infer_ms: args.inferMs,
  };
}

export const controlMessage = (action: ControlAction): ControlMsg => ({ v: 1, type: "control", action });
