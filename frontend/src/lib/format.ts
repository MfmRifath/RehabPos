// Display formatting shared across screens. Values come from the server; nothing is rounded here
// beyond what the screen shows.
import type { RepOut } from "../ws/protocol";

/** Degrees as a whole number with a degree sign; ratio-unit measures keep two decimals. */
export function formatMeasure(value: number | null | undefined, unit: string): string {
  if (value === null || value === undefined) return "–";
  return unit === "deg" ? `${Math.round(value)}°` : value.toFixed(2);
}

/** Share of the target reached, clamped to 0-1 (the UI never rewards going past the target). */
export function targetFraction(value: number | null | undefined, target: number): number {
  if (value === null || value === undefined || target <= 0) return 0;
  return Math.min(1, Math.max(0, value / target));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

/** Calendar date ("YYYY-MM-DD") as a short axis label, without shifting across time zones. */
export function shortDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  if (!y || !m || !d) return iso;
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export const plural = (n: number, one: string, many = `${one}s`): string => `${n} ${n === 1 ? one : many}`;

export const REP_LABELS: Record<RepOut["label"], string> = {
  correct: "Clean",
  incomplete_rom: "Partial range",
  too_fast: "Quick",
  compensated: "Form check",
};

export const viewLabel = (view: string): string => (view === "sagittal" ? "Side-on" : "Facing camera");
