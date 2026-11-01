import type { FeedbackMsg, RepOut } from "../ws/protocol";
import { initialLive, liveReducer, useApp } from "./app";

const fb: FeedbackMsg = {
  v: 1, type: "feedback", frame: 5, ts_ms: 50, angles: { upper_arm_trunk_angle: 90 }, phase: "peak",
  rep: { count: 2, last_peak: 150, last_duration_s: 5, last_similarity: 0.9 },
  flags: [{ code: "TRUNK_LEAN", severity: "warn", confidence: 0.9, source: "rule" }],
  cue: { code: "TRUNK_LEAN", text: "Keep your trunk upright", speak: true, severity: "warn" },
  quality: { visibility_ok: true, view_ok: false }, server_ms: 0.3, dropped: 4,
};
const rep = (idx: number, label: RepOut["label"]): RepOut => ({ idx, t_start: 0, t_peak: 1, t_end: 2, peak_angle: 150, duration_s: 5, peak_vel: 50, similarity: 0.9, label, flags: [] });

describe("liveReducer", () => {
  it("applies feedback", () => {
    const s = liveReducer(initialLive, fb, 12);
    expect(s).toMatchObject({ phase: "peak", repCount: 2, viewOk: false, serverMs: 0.3, rttMs: 12, dropped: 4 });
    expect(s.cue?.code).toBe("TRUNK_LEAN");
    expect(liveReducer(s, { ...fb, dropped: undefined }, null)).toMatchObject({ rttMs: 12, dropped: 4 });
  });
  it("records reps once and scores them", () => {
    let s = liveReducer(initialLive, { v: 1, type: "rep_completed", rep: rep(0, "correct") });
    s = liveReducer(s, { v: 1, type: "rep_completed", rep: rep(0, "correct") }); // duplicate after reconnect
    s = liveReducer(s, { v: 1, type: "rep_completed", rep: rep(1, "too_fast") });
    expect(s.reps.map((r) => r.idx)).toEqual([0, 1]);
    expect(s.game.points).toBe(12);
  });
  it("setup status and errors", () => {
    const s = liveReducer(initialLive, { v: 1, type: "setup_status", calibrated: true, visibility_ok: false, view_ok: true, hints: ["Face the camera"] });
    expect(s).toMatchObject({ calibrated: true, visibilityOk: false, hints: ["Face the camera"] });
    expect(liveReducer(s, { v: 1, type: "error", code: "bad_message", message: "lm" }).error).toBe("bad_message: lm");
  });
});

it("store actions: disclaimer persists, mute/hud toggle, resetLive", () => {
  const st = useApp.getState();
  expect(st.profile.userId).toMatch(/^u-[a-z0-9]+$/);
  st.acceptDisclaimer();
  expect(JSON.parse(localStorage.getItem("rehabpose.profile.v1") as string).disclaimerAccepted).toBe(true);
  st.toggleMute();
  st.toggleHud();
  expect(useApp.getState()).toMatchObject({ muted: true, showHud: true });
  st.applyMessage(fb, 5);
  st.resetLive("s9");
  expect(useApp.getState().live).toMatchObject({ sid: "s9", repCount: 0 });
});
