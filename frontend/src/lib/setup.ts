// Setup checks from landmarks: visibility, camera view orientation, distance.
// Thresholds mirror EngineConfig (frontal_ratio 0.45, sagittal_ratio 0.25) and the 0.6 gate.

export type View = "sagittal" | "frontal";
export interface Pt {
  x: number;
  y: number;
  visibility?: number;
}

export const VISIBILITY_THRESHOLD = 0.6;
export const UPPER_BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24];

export function visibleAll(lm: Pt[], indices: number[], thr = VISIBILITY_THRESHOLD): boolean {
  return indices.every((i) => (lm[i]?.visibility ?? 0) >= thr);
}

const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a: Pt, b: Pt, w: number, h: number): number => Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);

/** Frontal/sagittal from shoulder-width / torso-length (pixel space); null if ambiguous. */
export function detectView(lm: Pt[], w: number, h: number): View | null {
  const [ls, rs, lh, rh] = [lm[11], lm[12], lm[23], lm[24]];
  if (!ls || !rs || !lh || !rh) return null;
  const torso = dist(mid(ls, rs), mid(lh, rh), w, h);
  if (torso <= 0) return null;
  const ratio = dist(ls, rs, w, h) / torso;
  if (ratio >= 0.45) return "frontal";
  if (ratio <= 0.25) return "sagittal";
  return null;
}

export type DistanceHint = "ok" | "too_close" | "too_far";

/** Torso (mid-shoulder to mid-hip) as a fraction of image height: ~0.2-0.45 works well. */
export function distanceHint(lm: Pt[]): DistanceHint {
  const [ls, rs, lh, rh] = [lm[11], lm[12], lm[23], lm[24]];
  if (!ls || !rs || !lh || !rh) return "too_close";
  const frac = Math.abs(mid(lh, rh).y - mid(ls, rs).y);
  if (frac > 0.45) return "too_close";
  if (frac < 0.15) return "too_far";
  return "ok";
}

export interface SetupReport {
  personVisible: boolean;
  view: View | null;
  viewOk: boolean;
  distance: DistanceHint;
  ready: boolean;
}

export function setupReport(lm: Pt[] | null, w: number, h: number, expected: View): SetupReport {
  if (!lm) return { personVisible: false, view: null, viewOk: false, distance: "too_far", ready: false };
  const personVisible = visibleAll(lm, UPPER_BODY);
  const view = detectView(lm, w, h);
  const viewOk = view === null || view === expected;
  const distance = distanceHint(lm);
  return { personVisible, view, viewOk, distance, ready: personVisible && viewOk && distance === "ok" };
}
