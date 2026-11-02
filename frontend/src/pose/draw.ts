// Canvas overlay: skeleton, active joints, angle arc + value, target band.
// Drawn in mirrored coordinates (selfie view) so text stays readable.

export interface P {
  x: number;
  y: number;
  visibility?: number;
}

export const UPPER_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26],
];

const SIDE = { L: { s: 11, e: 13, w: 15, h: 23 }, R: { s: 12, e: 14, w: 16, h: 24 } } as const;

export interface OverlayOptions {
  side: "L" | "R";
  primary: string; // measure name
  angle: number | null; // degrees (from the server)
  target: number;
  romFraction?: number;
  unit: string;
  mirror: boolean;
  colors: { bone: string; active: string; band: string; text: string };
  visibilityThreshold?: number;
}

/** Overlay colours; the active colour matches --stage-accent. The stage is dark in both themes. */
export const OVERLAY_COLORS: OverlayOptions["colors"] = {
  bone: "rgba(255,255,255,0.85)",
  active: "#4fe0cf",
  band: "rgba(79,224,207,0.45)",
  text: "#fff",
};

export function drawOverlay(ctx: CanvasRenderingContext2D, lm: P[] | null, w: number, h: number, o: OverlayOptions): void {
  ctx.clearRect(0, 0, w, h);
  if (!lm) return;
  const thr = o.visibilityThreshold ?? 0.6;
  const px = (p: P): [number, number] => [(o.mirror ? 1 - p.x : p.x) * w, p.y * h];
  const ok = (i: number) => (lm[i]?.visibility ?? 0) >= thr;
  const j = SIDE[o.side];
  const active = new Set<number>([j.s, j.e, j.w, j.h]);

  ctx.lineCap = "round";
  for (const [a, b] of UPPER_CONNECTIONS) {
    const pa = lm[a];
    const pb = lm[b];
    if (!pa || !pb || !ok(a) || !ok(b)) continue;
    const isActive = active.has(a) && active.has(b);
    ctx.strokeStyle = isActive ? o.colors.active : o.colors.bone;
    ctx.lineWidth = isActive ? 5 : 3;
    ctx.beginPath();
    ctx.moveTo(...px(pa));
    ctx.lineTo(...px(pb));
    ctx.stroke();
  }
  for (const i of active) {
    const p = lm[i];
    if (!p || !ok(i)) continue;
    ctx.fillStyle = o.colors.active;
    ctx.beginPath();
    ctx.arc(...px(p), 7, 0, Math.PI * 2);
    ctx.fill();
  }

  // Angle arc for angle-type primaries at the vertex joint.
  const vertex = o.primary === "elbow_flexion" ? j.e : o.primary === "upper_arm_trunk_angle" ? j.s : null;
  if (vertex === null || o.angle === null || o.unit !== "deg") return;
  const ref = o.primary === "elbow_flexion" ? lm[j.s] : lm[j.h]; // 0 deg reference ray
  const v = lm[vertex];
  if (!v || !ref || !ok(vertex)) return;
  const [vx, vy] = px(v);
  const [rx, ry] = px(ref);
  // elbow_flexion = 180 - interior: reference is the extension of the upper arm (away from S).
  const refAngle = Math.atan2(ry - vy, rx - vx) + (o.primary === "elbow_flexion" ? Math.PI : 0);
  const moving = lm[o.primary === "elbow_flexion" ? j.w : j.e];
  if (!moving) return;
  const [mx, my] = px(moving);
  const dir = Math.sign(Math.sin(Math.atan2(my - vy, mx - vx) - refAngle)) || 1;
  const rad = (d: number) => (d * Math.PI) / 180;
  const r = Math.max(28, Math.min(w, h) * 0.07);

  const frac = o.romFraction ?? 0.9;
  ctx.strokeStyle = o.colors.band;
  ctx.lineWidth = 10;
  ctx.beginPath();
  ctx.arc(vx, vy, r, refAngle + dir * rad(o.target * frac), refAngle + dir * rad(o.target), dir < 0);
  ctx.stroke();

  ctx.strokeStyle = o.colors.active;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(vx, vy, r * 0.75, refAngle, refAngle + dir * rad(o.angle), dir < 0);
  ctx.stroke();

  const label = `${Math.round(o.angle)}°`;
  // Canvas pixels scale with the stage, so the label is sized from the frame height.
  const size = Math.max(20, Math.round(h * 0.05));
  ctx.font = `800 ${size}px "Nunito Variable", system-ui, sans-serif`;
  const tx = vx + r + 8;
  const ty = vy - r * 0.6;
  const m = ctx.measureText(label);
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(tx - size * 0.3, ty - size * 0.95, m.width + size * 0.6, size * 1.25);
  ctx.fillStyle = o.colors.text;
  ctx.fillText(label, tx, ty);
}
