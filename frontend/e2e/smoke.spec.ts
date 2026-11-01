import { expect, test } from "@playwright/test";

const exercise = {
  id: "shoulder_flexion", name: "Shoulder flexion", view: "sagittal", allowed_views: ["sagittal"],
  experimental: false, primary_angle: "upper_arm_trunk_angle", unit: "deg", target: 150, min_rep_s: 2,
  fsm: { start_below: 30, peak_above_offset: 15, return_below: 40 },
  rules: [{ code: "TRUNK_LEAN", kind: "frame", measure: "trunk_tilt", op: "abs_delta_gt", value: 10, severity: "warn", experimental: false }],
  cue_texts: { TRUNK_LEAN: "Keep your trunk upright" },
};
const session = {
  id: "s-test", user_id: "u-x", exercise_id: "shoulder_flexion", side: "R", view: "sagittal", source: "live",
  started_at: "2026-11-01T10:00:00Z", ended_at: null, rep_count: 0, mean_rom: null, quality_score: null, model_version: "test",
};
const feedback = {
  v: 1, type: "feedback", frame: 0, ts_ms: 1, angles: { upper_arm_trunk_angle: 97 }, phase: "ascending",
  rep: { count: 1, last_peak: 150, last_duration_s: 5, last_similarity: 0.9 },
  flags: [{ code: "TRUNK_LEAN", severity: "warn", confidence: 0.9, source: "rule" }],
  cue: { code: "TRUNK_LEAN", text: "Keep your trunk upright", speak: true, severity: "warn" },
  quality: { visibility_ok: true, view_ok: true }, server_ms: 0.2, dropped: 0,
};
const rep = { idx: 0, t_start: 1, t_peak: 3, t_end: 6, peak_angle: 150, duration_s: 5, peak_vel: 70, similarity: 0.9, label: "correct", flags: [] };

test("disclaimer -> exercise -> setup -> live (mocked WS) -> summary, all assets local", async ({ page, baseURL }) => {
  const external: string[] = [];
  page.on("request", (r) => {
    if (!r.url().startsWith(baseURL as string) && !r.url().startsWith("data:") && !r.url().startsWith("blob:")) external.push(r.url());
  });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const cspBlocked: string[] = [];
  page.on("console", (m) => {
    if (m.text().includes("Content Security Policy")) cspBlocked.push(m.text());
  });

  await page.route("**/api/exercises", (r) => r.fulfill({ json: [exercise] }));
  await page.route("**/api/sessions", (r) => r.fulfill({ status: 201, json: session }));
  await page.route("**/api/sessions/s-test/end", (r) => r.fulfill({ json: { ...session, ended_at: "2026-11-01T10:05:00Z", rep_count: 1 } }));
  await page.route("**/api/sessions/s-test/report.json", (r) =>
    r.fulfill({ json: { summary: { rep_count: 1, mean_rom: 150, best_rom: 150, quality_score: 1, labels: { correct: 1 } }, target: { rom: 150, unit: "deg" }, reps: [{ ...rep, peak: 150 }], flags_summary: {}, disclaimer: "x" } }),
  );
  const fromClient: { type: string; action?: string }[] = [];
  await page.routeWebSocket("**/ws/session/**", (ws) => {
    ws.onMessage((m) => fromClient.push(JSON.parse(String(m))));
    ws.send(JSON.stringify({ v: 1, type: "setup_status", calibrated: true, visibility_ok: true, view_ok: true, hints: [] }));
    ws.send(JSON.stringify(feedback));
    ws.send(JSON.stringify({ v: 1, type: "rep_completed", rep }));
  });

  await page.goto("/");
  await expect(page.getByTestId("disclaimer-text")).toContainText("It is not a medical device");
  await page.getByRole("button", { name: "I understand" }).click();
  await expect(page.getByTestId("exercise-shoulder_flexion")).toBeVisible();

  await page.getByTestId("go-setup").click();
  await expect(page.getByRole("heading", { name: "Setup check" })).toBeVisible();
  // The fake camera shows no person, so the pose model loads but setup is not "ready".
  await expect(page.getByText(/Pose model ready \((GPU|CPU)\)/)).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "Start anyway" }).click();

  await expect(page.getByTestId("status")).toContainText("Session s-test");
  await expect(page.getByTestId("cue")).toContainText("Keep your trunk upright");
  await expect(page.getByTestId("cue")).toHaveClass(/warn/);
  await expect(page.getByTestId("rep-count")).toHaveText("1");
  await expect(page.getByText("10 pts · combo 1")).toBeVisible();

  await page.keyboard.press("h");
  await expect(page.getByTestId("hud")).toContainText("server 0.20 ms");
  await page.keyboard.press("m");
  await expect(page.getByRole("button", { name: "Unmute (m)" })).toBeVisible();
  await page.keyboard.press(" ");
  await expect.poll(() => fromClient.some((m) => m.type === "control" && m.action === "pause")).toBe(true);

  await page.getByTestId("end").click();
  await expect(page.getByRole("heading", { name: "Session summary" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download PDF report" })).toHaveAttribute("href", "/api/sessions/s-test/report.pdf");

  expect(external, "no runtime CDN / external requests").toEqual([]);
  expect(errors).toEqual([]);
  // MediaPipe's telemetry attempt is refused by the CSP (and nothing else is).
  for (const msg of cspBlocked) expect(msg).toContain("odml.pa.googleapis.com");
  console.log(`CSP-blocked requests: ${cspBlocked.length}`);
});

test("production page ships a same-origin-only Content-Security-Policy", async ({ page }) => {
  await page.route("**/api/exercises", (r) => r.fulfill({ json: [exercise] }));
  await page.goto("/");
  const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content");
  expect(csp).toContain("connect-src 'self'");
  expect(csp).toContain("default-src 'self'");
  // The exact telemetry call @mediapipe/tasks-vision makes is refused by the browser.
  const outcome = await page.evaluate(() =>
    fetch("https://odml.pa.googleapis.com/v1/log", { method: "POST", body: "x" }).then(
      () => "sent",
      () => "blocked",
    ),
  );
  expect(outcome).toBe("blocked");
});

test("progress shows server badges; delete asks first and resets the user", async ({ page }) => {
  const series = [{ date: "2026-03-10", sessions: 2, reps: 60, mean_rom: 140, quality: 0.8 }];
  const progress = {
    user_id: "u-x", today: "2026-03-10",
    overall: { sessions: 2, total_reps: 60, active_days: 1, current_streak: 1, longest_streak: 1 },
    badges: [
      { id: "first_50", label: "First 50 reps", earned: true, detail: "60 / 50 reps" },
      { id: "streak_7", label: "7-day streak", earned: false, detail: "longest streak 1 days, current 1" },
    ],
    exercises: { shoulder_flexion: { sessions: 2, total_reps: 60, best_mean_rom: 140, rom_personal_best: { value: 141, date: "2026-03-10" }, series } },
  };
  const urls: string[] = [];
  const deletes: string[] = [];
  await page.route("**/api/exercises", (r) => r.fulfill({ json: [exercise] }));
  await page.route("**/api/users/**", (r) => {
    urls.push(r.request().url());
    if (r.request().method() === "DELETE") {
      deletes.push(r.request().url());
      return r.fulfill({ json: { user_id: "u-x", deleted: { sessions: 2, reps: 60 } } });
    }
    return r.fulfill({ json: progress });
  });
  await page.goto("/#/progress");
  await page.getByRole("button", { name: "I understand" }).click();
  await expect(page.getByTestId("overall")).toContainText("60 reps in 2 sessions");
  await expect(page.getByText("🏅 First 50 reps")).toBeVisible();
  await expect(page.getByText("○ 7-day streak")).toBeVisible();
  await expect(page.getByText("best session 141° on 2026-03-10")).toBeVisible();
  expect(urls[0]).toContain("tz_offset_minutes=");

  page.once("dialog", (d) => d.dismiss());
  await page.getByTestId("delete-data").click();
  expect(deletes).toEqual([]); // cancelled: nothing deleted

  const before = await page.evaluate(() => localStorage.getItem("rehabpose.profile.v1"));
  page.once("dialog", (d) => d.accept());
  await page.getByTestId("delete-data").click();
  await expect.poll(() => deletes.length).toBe(1);
  await expect.poll(() => page.evaluate(() => localStorage.getItem("rehabpose.profile.v1"))).not.toBe(before);
});
