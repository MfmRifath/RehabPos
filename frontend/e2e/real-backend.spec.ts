import { expect, test } from "@playwright/test";

// Runs against a real `rehabpose serve` that serves frontend/dist (no mocks).
// Enable with E2E_BACKEND_URL=http://127.0.0.1:8000 npx playwright test e2e/real-backend.spec.ts
const backend = process.env.E2E_BACKEND_URL;
test.skip(!backend, "set E2E_BACKEND_URL to run against a live backend");

test("real backend: exercises load, live session connects over WS", async ({ page }) => {
  const external: string[] = [];
  page.on("request", (r) => {
    const u = r.url();
    if (!u.startsWith(backend as string) && !u.startsWith("ws") && !u.startsWith("data:") && !u.startsWith("blob:")) external.push(u);
  });
  await page.goto(`${backend}/`);
  await page.getByRole("button", { name: "I understand" }).click();
  await expect(page.getByTestId("exercise-shoulder_abduction")).toBeVisible();
  await page.getByTestId("exercise-shoulder_abduction").click();
  await page.getByTestId("go-setup").click();
  await expect(page.getByText(/Pose model ready \((GPU|CPU)\)/)).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Start anyway" }).click();
  await expect(page.getByTestId("status")).toContainText(/Session [0-9a-f]{16}/);
  await page.keyboard.press("h");
  await expect(page.getByTestId("hud")).toContainText("ws open", { timeout: 10_000 });
  // Landmarker is shared from the setup page: no second model load.
  await expect(page.getByTestId("hud")).toContainText(/pose [0-9.]+ ms \((GPU|CPU)\)/, { timeout: 10_000 });
  // The pose loop must be processing camera frames (fake camera; this is not a performance number).
  await expect(page.getByTestId("hud")).toContainText(/FPS [1-9][0-9.]*/, { timeout: 10_000 });
  const fps = await page.getByTestId("hud").textContent();
  console.log(`HUD (headless Chrome, fake camera): ${fps?.replace(/\s+/g, " ")}`);
  await page.getByTestId("end").click();
  await expect(page.getByRole("heading", { name: "Session summary" })).toBeVisible();
  expect(external).toEqual([]);
});
