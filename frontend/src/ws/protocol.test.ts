import { controlMessage, landmarksMessage, parseServerMessage } from "./protocol";

const lm = Array.from({ length: 33 }, (_, i) => ({ x: i / 100, y: 0.5, z: 0, visibility: 0.9 }));
const world = Array.from({ length: 33 }, () => ({ x: 0.1, y: -0.2, z: 0.3, visibility: 0 }));

const feedback = {
  v: 1, type: "feedback", frame: 7, ts_ms: 100, angles: { upper_arm_trunk_angle: 42 }, phase: "ascending",
  rep: { count: 1, last_peak: 150, last_duration_s: 5, last_similarity: 0.9 }, flags: [], cue: null,
  quality: { visibility_ok: true, view_ok: true }, server_ms: 0.2,
};

describe("parseServerMessage", () => {
  it("accepts every v1 server message type", () => {
    expect(parseServerMessage(JSON.stringify(feedback))?.type).toBe("feedback");
    expect(parseServerMessage(JSON.stringify({ v: 1, type: "rep_completed", rep: { idx: 0 } }))?.type).toBe("rep_completed");
    expect(parseServerMessage(JSON.stringify({ v: 1, type: "setup_status", calibrated: true, visibility_ok: true, view_ok: true, hints: [] }))?.type).toBe("setup_status");
    expect(parseServerMessage(JSON.stringify({ v: 1, type: "error", code: "bad_message", message: "x" }))?.type).toBe("error");
  });

  it.each([
    "not json",
    JSON.stringify({ ...feedback, v: 2 }),
    JSON.stringify({ ...feedback, type: "unknown" }),
    JSON.stringify({ ...feedback, rep: undefined }),
    JSON.stringify({ v: 1, type: "rep_completed", rep: {} }),
    JSON.stringify({ v: 1, type: "error" }),
    JSON.stringify([1, 2]),
  ])("rejects malformed input %#", (raw) => {
    expect(parseServerMessage(raw)).toBeNull();
  });
});

describe("landmarksMessage", () => {
  it("builds a v1 message and copies image visibility into world landmarks", () => {
    const m = landmarksMessage({ sid: "s1", frame: 3, tsMs: 1234.6, width: 640, height: 480, landmarks: lm, world, inferMs: 9 });
    expect(m).toMatchObject({ v: 1, type: "landmarks", sid: "s1", frame: 3, ts_ms: 1235, img: { w: 640, h: 480 }, client_infer_ms: 9 });
    expect(m.lm).toHaveLength(33);
    expect(m.lm[5]).toEqual([0.05, 0.5, 0, 0.9]);
    expect(m.world[5]).toEqual([0.1, -0.2, 0.3, 0.9]);
  });

  it("rejects a wrong landmark count", () => {
    expect(() => landmarksMessage({ sid: "s", frame: 0, tsMs: 0, width: 1, height: 1, landmarks: lm.slice(0, 17), world })).toThrow(/33/);
  });

  it("defaults missing visibility to 0", () => {
    const noVis = lm.map(({ x, y, z }) => ({ x, y, z }));
    expect(landmarksMessage({ sid: "s", frame: 0, tsMs: 0, width: 1, height: 1, landmarks: noVis, world }).lm[0]?.[3]).toBe(0);
  });
});

it("controlMessage", () => {
  expect(controlMessage("pause")).toEqual({ v: 1, type: "control", action: "pause" });
});

describe("contract with the Python backend (protocol.fixtures.json)", () => {
  it("parses every server message the backend produces", async () => {
    const fx = (await import("./protocol.fixtures.json")).default as { server: unknown[]; client: unknown[] };
    for (const m of fx.server) expect(parseServerMessage(JSON.stringify(m))).toEqual(m);
  });
  it("builds client messages identical to the backend's accepted samples", async () => {
    const fx = (await import("./protocol.fixtures.json")).default as { client: unknown[] };
    const built = landmarksMessage({
      sid: "s1", frame: 3, tsMs: 1234.6, width: 640, height: 480, inferMs: 9,
      landmarks: Array.from({ length: 33 }, (_, i) => ({ x: Math.round(i) / 100, y: 0.5, z: 0, visibility: 0.9 })),
      world: Array.from({ length: 33 }, () => ({ x: 0.1, y: -0.2, z: 0.3, visibility: 0 })),
    });
    expect(built).toEqual(fx.client[0]);
    expect(controlMessage("pause")).toEqual(fx.client[1]);
  });
});
