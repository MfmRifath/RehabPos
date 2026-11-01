import { drawOverlay, type P } from "./draw";

function recorder() {
  const calls: string[] = [];
  const ctx = new Proxy(
    { measureText: () => ({ width: 30 }) } as Record<string, unknown>,
    {
      get(target, prop: string) {
        if (prop in target) return target[prop];
        return (...args: unknown[]) => calls.push(`${prop}(${args.map((a) => (typeof a === "number" ? a.toFixed(0) : String(a))).join(",")})`);
      },
      set() {
        return true;
      },
    },
  );
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
}

const pose = (vis = 0.9): P[] => {
  const p: P[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: vis }));
  p[12] = { x: 0.4, y: 0.3, visibility: vis }; // R shoulder
  p[14] = { x: 0.4, y: 0.5, visibility: vis }; // R elbow (arm down)
  p[16] = { x: 0.4, y: 0.7, visibility: vis };
  p[24] = { x: 0.42, y: 0.8, visibility: vis };
  return p;
};

const opts = { side: "R" as const, primary: "upper_arm_trunk_angle", angle: 30, target: 150, unit: "deg", mirror: true, colors: { bone: "w", active: "y", band: "g", text: "t" } };

it("draws skeleton, active joints, target band, arc and angle label", () => {
  const { ctx, calls } = recorder();
  drawOverlay(ctx, pose(), 1000, 1000, opts);
  expect(calls[0]).toBe("clearRect(0,0,1000,1000)");
  expect(calls.filter((c) => c.startsWith("lineTo")).length).toBeGreaterThan(5);
  expect(calls.filter((c) => c.startsWith("arc(")).length).toBe(4 + 2); // 4 joints + band + angle arc
  expect(calls.some((c) => c.startsWith("fillText(30°"))).toBe(true);
  // mirrored: right shoulder at x=0.4 is drawn at x=600
  expect(calls).toContain("arc(600,300,7,0,6)");
});

it("skips invisible joints and the arc when the angle is unknown", () => {
  const { ctx, calls } = recorder();
  drawOverlay(ctx, pose(0.1), 1000, 1000, opts);
  expect(calls).toEqual(["clearRect(0,0,1000,1000)"]);
  const r2 = recorder();
  drawOverlay(r2.ctx, pose(), 1000, 1000, { ...opts, angle: null });
  expect(r2.calls.some((c) => c.startsWith("fillText"))).toBe(false);
  const r3 = recorder();
  drawOverlay(r3.ctx, null, 10, 10, opts);
  expect(r3.calls).toEqual(["clearRect(0,0,10,10)"]);
});
