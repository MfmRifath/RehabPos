import { ApiError } from "./api";
import { describeApiError, describeCameraError } from "./errors";
import { formatBytes, formatMeasure, plural, shortDate, targetFraction } from "./format";
import { setupChecklist, setupReport, type Pt } from "./setup";

const named = (name: string, message = "x") => Object.assign(new Error(message), { name });

describe("describeCameraError", () => {
  it.each([
    ["NotAllowedError", "Camera access is blocked"],
    ["SecurityError", "Camera access is blocked"],
    ["NotFoundError", "No camera found"],
    ["OverconstrainedError", "No camera found"],
    ["NotReadableError", "The camera is busy"],
  ])("%s", (name, title) => {
    expect(describeCameraError(named(name)).title).toBe(title);
  });

  it("recognises an insecure page and a failed model", () => {
    expect(describeCameraError(new Error("Camera API unavailable (the page must be...)")).title).toBe("The camera needs a secure page");
    expect(describeCameraError(new Error("could not create PoseLandmarker: boom")).title).toBe("The pose model did not load");
  });

  it("falls back to the original message", () => {
    expect(describeCameraError("odd failure")).toEqual({ title: "The camera did not start", detail: "odd failure" });
  });
});

describe("describeApiError", () => {
  it("prefers the server's reason", () => {
    expect(describeApiError(new ApiError(409, "session in use"))).toBe("session in use");
    expect(describeApiError(new ApiError(500, ""))).toContain("500");
  });

  it("explains a network failure", () => {
    expect(describeApiError(new TypeError("Failed to fetch"))).toContain("Cannot reach the RehabPose server");
  });
});

describe("format", () => {
  it("formatMeasure uses whole degrees or a two-decimal ratio", () => {
    expect(formatMeasure(97.4, "deg")).toBe("97°");
    expect(formatMeasure(0.756, "ratio")).toBe("0.76");
    expect(formatMeasure(null, "deg")).toBe("–");
  });

  it("targetFraction is clamped and safe with a zero target", () => {
    expect(targetFraction(75, 150)).toBe(0.5);
    expect(targetFraction(300, 150)).toBe(1);
    expect(targetFraction(-5, 150)).toBe(0);
    expect(targetFraction(10, 0)).toBe(0);
    expect(targetFraction(null, 150)).toBe(0);
  });

  it("formatBytes picks a readable unit", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(5 * 1024 ** 2)).toBe("5.0 MB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.00 GB");
  });

  it("shortDate keeps the calendar day and tolerates bad input", () => {
    expect(shortDate("2026-03-10")).toContain("10");
    expect(shortDate("not-a-date")).toBe("not-a-date");
  });

  it("plural", () => {
    expect(plural(1, "day")).toBe("1 day");
    expect(plural(2, "day")).toBe("2 days");
    expect(plural(0, "rep")).toBe("0 reps");
  });
});

describe("setupChecklist", () => {
  const byId = (checks: ReturnType<typeof setupChecklist>) => Object.fromEntries(checks.map((c) => [c.id, c]));

  it("is all pending before the first frame", () => {
    expect(setupChecklist(null, "sagittal", "R").every((c) => c.state === "pending")).toBe(true);
  });

  it("asks the person to step in, and does not fail checks it cannot measure", () => {
    const c = byId(setupChecklist(setupReport(null, 1280, 720, "sagittal", "R"), "sagittal", "R"));
    expect(c.person?.state).toBe("problem");
    expect(c.person?.text).toBe("Step into the camera's view");
    expect(c.joints?.state).toBe("pending");
    expect(c.view?.state).toBe("pending");
    expect(c.distance?.state).toBe("pending");
    expect(c.distance?.text).not.toContain("good");
  });

  it("reports every check once a person is visible", () => {
    const lm: Pt[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
    lm[11] = { x: 0.5, y: 0.3, visibility: 0.9 };
    lm[12] = { x: 0.52, y: 0.3, visibility: 0.9 };
    lm[23] = { x: 0.5, y: 0.6, visibility: 0.9 };
    lm[24] = { x: 0.52, y: 0.6, visibility: 0.9 };
    const checks = setupChecklist(setupReport(lm, 1280, 720, "sagittal", "R"), "sagittal", "R");
    expect(checks.every((c) => c.state === "ok")).toBe(true);
    expect(byId(checks).distance?.text).toBe("Distance looks good");
  });

  it("names missing joints and the way to fix the distance", () => {
    const lm: Pt[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
    lm[11] = { x: 0.5, y: 0.1, visibility: 0.9 };
    lm[12] = { x: 0.52, y: 0.1, visibility: 0.9 };
    lm[23] = { x: 0.5, y: 0.9, visibility: 0.9 };
    lm[24] = { x: 0.52, y: 0.9, visibility: 0.9 };
    lm[16] = { x: 0.5, y: 0.5, visibility: 0.1 };
    const c = byId(setupChecklist(setupReport(lm, 1280, 720, "sagittal", "R"), "sagittal", "R"));
    expect(c.joints?.text).toBe("Not clearly visible: right wrist");
    expect(c.joints?.state).toBe("problem");
    expect(c.distance?.text).toBe("Step back a little");
  });
});
