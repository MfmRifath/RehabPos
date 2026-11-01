import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { DISCLAIMER } from "./disclaimer";
import { gameReducer, initialGame } from "./gamification";
import { detectView, distanceHint, missingJoints, requiredJoints, setupReport, visibleAll, type Pt } from "./setup";
import { CueSpeaker } from "./speech";

describe("CueSpeaker", () => {
  const cue = { code: "TRUNK_LEAN", text: "Keep your trunk upright", speak: true };
  it("speaks at most once per 3 s and respects mute and speak=false", () => {
    const say = vi.fn();
    const s = new CueSpeaker(3000, say);
    expect(s.handle(cue, 0, false)).toBe(true);
    expect(s.handle(cue, 2999, false)).toBe(false);
    expect(s.handle(cue, 3000, false)).toBe(true);
    expect(s.handle(cue, 9000, true)).toBe(false);
    expect(s.handle({ ...cue, speak: false }, 9000, false)).toBe(false);
    expect(s.handle(null, 9000, false)).toBe(false);
    expect(say).toHaveBeenCalledTimes(2);
    expect(say).toHaveBeenCalledWith("Keep your trunk upright");
  });
});

describe("gameReducer", () => {
  it("rewards clean reps with growing combos and resets on others", () => {
    let g = initialGame;
    g = gameReducer(g, "correct");
    expect(g).toMatchObject({ points: 10, combo: 1, lastGain: 10 });
    g = gameReducer(g, "correct");
    expect(g).toMatchObject({ points: 22, combo: 2, bestCombo: 2, cleanReps: 2 });
    g = gameReducer(g, "compensated");
    expect(g).toMatchObject({ points: 24, combo: 0, bestCombo: 2, cleanReps: 2, lastGain: 2 });
  });
  it("caps the combo bonus", () => {
    let g = initialGame;
    for (let i = 0; i < 30; i++) g = gameReducer(g, "correct");
    expect(g.lastGain).toBe(10 + 2 * 10);
  });
});

// Frontal skeleton in normalized coords (shoulders 0.3 wide, torso 0.3 tall) on a 1000x1000 image.
function skeleton(shoulderHalf = 0.15, torso = 0.3, vis = 0.9): Pt[] {
  const p = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: vis }));
  p[11] = { x: 0.5 + shoulderHalf, y: 0.3, visibility: vis };
  p[12] = { x: 0.5 - shoulderHalf, y: 0.3, visibility: vis };
  p[23] = { x: 0.6, y: 0.3 + torso, visibility: vis };
  p[24] = { x: 0.4, y: 0.3 + torso, visibility: vis };
  return p;
}

describe("setup heuristics", () => {
  it("detects frontal / sagittal / ambiguous views", () => {
    expect(detectView(skeleton(0.15), 1000, 1000)).toBe("frontal"); // ratio 1.0
    expect(detectView(skeleton(0.02), 1000, 1000)).toBe("sagittal"); // ratio ~0.13
    expect(detectView(skeleton(0.05), 1000, 1000)).toBeNull(); // ratio ~0.33
  });
  it("distance hints from torso height", () => {
    expect(distanceHint(skeleton(0.15, 0.3))).toBe("ok");
    expect(distanceHint(skeleton(0.15, 0.6))).toBe("too_close");
    expect(distanceHint(skeleton(0.15, 0.1))).toBe("too_far");
  });
  it("visibility gate and overall readiness", () => {
    expect(visibleAll(skeleton(), [11, 12])).toBe(true);
    expect(visibleAll(skeleton(0.15, 0.3, 0.5), [11])).toBe(false);
    expect(setupReport(skeleton(), 1000, 1000, "frontal", "R").ready).toBe(true);
    expect(setupReport(skeleton(), 1000, 1000, "sagittal", "R")).toMatchObject({ ready: false, viewOk: false });
    expect(setupReport(null, 1, 1, "frontal", "R")).toMatchObject({ ready: false, detected: false });
  });
  it("side-on needs only the near arm and hip (regression: both arms were required)", () => {
    expect(requiredJoints("R", "sagittal")).toEqual([12, 14, 16, 24]);
    expect(requiredJoints("L", "sagittal")).toEqual([11, 13, 15, 23]);
    const lm = skeleton(0.02); // shoulders overlap: side-on
    for (const i of [0, 7, 11, 13, 15, 23]) (lm[i] as Pt).visibility = 0.1; // far side + face hidden
    expect(setupReport(lm, 1000, 1000, "sagittal", "R")).toMatchObject({ detected: true, personVisible: true, missing: [], ready: true });
    expect(setupReport(lm, 1000, 1000, "sagittal", "L")).toMatchObject({ personVisible: false, ready: false });
    expect(missingJoints(lm, "L", "sagittal")).toEqual(["left shoulder", "left elbow", "left wrist", "left hip"]);
  });
});

it("frontend disclaimer matches the Python package verbatim", () => {
  const py = readFileSync(resolve(__dirname, "../../../src/rehabpose/__init__.py"), "utf8");
  const block = py.slice(py.indexOf("DISCLAIMER = ("), py.indexOf(")", py.indexOf("DISCLAIMER = (")));
  const pyText = [...block.matchAll(/"([^"]*)"/g)].map((m) => m[1]).join("");
  expect(DISCLAIMER).toBe(pyText);
});
