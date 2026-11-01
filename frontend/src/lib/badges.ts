// Streaks and badges from the progress series (client-side until phase 8 moves them server-side).

export interface DayPoint {
  date: string; // YYYY-MM-DD
  sessions: number;
  reps: number;
  mean_rom: number | null;
  quality: number | null;
}

export interface Badge {
  id: string;
  label: string;
  earned: boolean;
  detail: string;
}

const DAY = 86_400_000;
const toDay = (d: string): number => Date.parse(`${d}T00:00:00Z`) / DAY;

/** Consecutive active days ending at `today` (or yesterday, so a streak survives until midnight). */
export function currentStreak(days: string[], today: string): number {
  const set = new Set(days.map(toDay));
  let d = toDay(today);
  if (!set.has(d)) d -= 1;
  let n = 0;
  while (set.has(d)) {
    n += 1;
    d -= 1;
  }
  return n;
}

export function longestStreak(days: string[]): number {
  const sorted = [...new Set(days.map(toDay))].sort((a, b) => a - b);
  let best = 0;
  let run = 0;
  sorted.forEach((d, i) => {
    run = i > 0 && d - (sorted[i - 1] as number) === 1 ? run + 1 : 1;
    best = Math.max(best, run);
  });
  return best;
}

export function badges(series: DayPoint[], today: string): Badge[] {
  const totalReps = series.reduce((s, d) => s + d.reps, 0);
  const days = series.filter((d) => d.sessions > 0).map((d) => d.date);
  const longest = longestStreak(days);
  const roms = series.map((d) => d.mean_rom).filter((v): v is number => v !== null);
  const last = roms.at(-1);
  const prevBest = roms.length > 1 ? Math.max(...roms.slice(0, -1)) : null;
  const pb = last !== undefined && prevBest !== null && last > prevBest;
  return [
    { id: "first_50", label: "First 50 reps", earned: totalReps >= 50, detail: `${totalReps} / 50 reps` },
    { id: "streak_7", label: "7-day streak", earned: longest >= 7, detail: `longest streak ${longest} days, current ${currentStreak(days, today)}` },
    {
      id: "rom_pb",
      label: "ROM personal best",
      earned: pb,
      detail: pb ? `latest day ${last?.toFixed(1)} beats ${prevBest?.toFixed(1)}` : "beat your best daily mean ROM",
    },
  ];
}
