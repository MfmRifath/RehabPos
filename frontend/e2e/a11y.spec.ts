import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// Scans every screen in both colour schemes with mocked data (no backend needed).
const exercise = {
  id: "shoulder_flexion", name: "Shoulder flexion", view: "sagittal", allowed_views: ["sagittal", "frontal"],
  experimental: false, primary_angle: "upper_arm_trunk_angle", unit: "deg", target: 150, min_rep_s: 2,
  fsm: { start_below: 30, peak_above_offset: 15, return_below: 40 },
  rules: [{ code: "TRUNK_LEAN", kind: "frame", measure: "trunk_tilt", op: "abs_delta_gt", value: 10, severity: "warn", experimental: true }],
  cue_texts: { TRUNK_LEAN: "Keep your trunk upright" },
};
const session = {
  id: "s-test", user_id: "u-x", exercise_id: "shoulder_flexion", side: "R", view: "sagittal", source: "live",
  started_at: "2026-11-01T10:00:00Z", ended_at: null, rep_count: 0, mean_rom: null, quality_score: null, model_version: "test",
};
const rep = { idx: 0, t_start: 1, t_end: 6, peak: 150, duration_s: 5, peak_vel: 70, similarity: 0.9, label: "correct", flags: ["TRUNK_LEAN"] };
const progress = {
  user_id: "u-x", today: "2026-03-10",
  overall: { sessions: 2, total_reps: 60, active_days: 1, current_streak: 1, longest_streak: 1 },
  badges: [
    { id: "first_50", label: "First 50 reps", earned: true, detail: "60 / 50 reps" },
    { id: "streak_7", label: "7-day streak", earned: false, detail: "longest streak 1 days, current 1" },
  ],
  exercises: {
    shoulder_flexion: {
      sessions: 2, total_reps: 60, best_mean_rom: 140, rom_personal_best: { value: 141, date: "2026-03-10" },
      series: [{ date: "2026-03-10", sessions: 2, reps: 60, mean_rom: 140, quality: 0.8 }],
    },
  },
};
const feedback = {
  v: 1, type: "feedback", frame: 0, ts_ms: 1, angles: { upper_arm_trunk_angle: 97 }, phase: "ascending",
  rep: { count: 1, last_peak: 150, last_duration_s: 5, last_similarity: 0.9 },
  flags: [{ code: "TRUNK_LEAN", severity: "warn", confidence: 0.9, source: "rule" }],
  cue: { code: "TRUNK_LEAN", text: "Keep your trunk upright", speak: false, severity: "warn" },
  quality: { visibility_ok: true, view_ok: true }, server_ms: 0.2, dropped: 0,
};

async function mockApi(page: Page) {
  await page.route("**/api/exercises", (r) => r.fulfill({ json: [exercise] }));
  await page.route("**/api/sessions", (r) => r.fulfill({ status: 201, json: session }));
  await page.route("**/api/sessions/s-test/report.json", (r) =>
    r.fulfill({ json: { session: { exercise_id: "shoulder_flexion", exercise_name: "Shoulder flexion" }, summary: { rep_count: 1, mean_rom: 150, best_rom: 150, quality_score: 1, labels: { correct: 1 } }, target: { rom: 150, unit: "deg" }, reps: [rep], flags_summary: {}, disclaimer: "x" } }),
  );
  await page.route("**/api/users/**", (r) => {
    if (r.request().url().includes("/targets")) return r.fulfill({ json: { user_id: "u-x", targets: [] } });
    return r.fulfill({ json: progress });
  });
  await page.routeWebSocket("**/ws/session/**", (ws) => {
    ws.onMessage(() => undefined);
    ws.send(JSON.stringify({ v: 1, type: "setup_status", calibrated: true, visibility_ok: true, view_ok: true, hints: [] }));
    ws.send(JSON.stringify(feedback));
  });
}

async function acceptDisclaimer(page: Page) {
  await page.addInitScript(() => localStorage.setItem("rehabpose.profile.v1", JSON.stringify({ userId: "u-x", disclaimerAccepted: true })));
}

async function scan(page: Page) {
  const { violations } = await new AxeBuilder({ page }).analyze();
  const serious = violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
}

const SCREENS: { name: string; path: string; ready: (page: Page) => Promise<void> }[] = [
  { name: "home", path: "/#/", ready: (p) => expect(p.getByTestId("exercise-shoulder_flexion")).toBeVisible() },
  { name: "setup", path: "/#/setup", ready: (p) => expect(p.getByText(/Pose model ready \((GPU|CPU)\)/)).toBeVisible({ timeout: 20_000 }) },
  { name: "live", path: "/#/live", ready: (p) => expect(p.getByTestId("cue")).toBeVisible({ timeout: 20_000 }) },
  { name: "summary", path: "/#/summary/s-test", ready: (p) => expect(p.getByRole("heading", { name: "Repetitions" })).toBeVisible() },
  { name: "progress", path: "/#/progress", ready: (p) => expect(p.getByTestId("overall")).toBeVisible() },
  { name: "upload", path: "/#/upload", ready: (p) => expect(p.getByRole("heading", { name: "Analyze a video" })).toBeVisible() },
  { name: "therapist", path: "/#/therapist", ready: (p) => expect(p.getByText("Keep your trunk upright")).toBeVisible() },
];

for (const colorScheme of ["light", "dark"] as const) {
  test.describe(colorScheme, () => {
    test.use({ colorScheme });

    test("disclaimer modal", async ({ page }) => {
      await mockApi(page);
      await page.goto("/");
      await expect(page.getByTestId("disclaimer-text")).toBeVisible();
      await scan(page);
    });

    for (const screen of SCREENS) {
      test(screen.name, async ({ page }) => {
        await mockApi(page);
        await acceptDisclaimer(page);
        await page.goto(screen.path);
        await screen.ready(page);
        await scan(page);
      });
    }
  });
}

test("a deep link to the live screen waits for the disclaimer", async ({ page }) => {
  await mockApi(page);
  const sessions: string[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/sessions")) sessions.push(r.url());
  });
  await page.goto("/#/live");
  await expect(page.getByTestId("disclaimer-text")).toBeVisible();
  await expect(page.getByTestId("exercise-shoulder_flexion")).toBeVisible();
  expect(sessions).toEqual([]);
  await page.getByRole("button", { name: "I understand" }).click();
  await expect(page.getByTestId("status")).toContainText("Session s-test");
});
