// Points and combos for clean reps — a small motivation layer ("gamified rehab").
import type { RepOut } from "../ws/protocol";

export interface GameState {
  points: number;
  combo: number;
  bestCombo: number;
  cleanReps: number;
  lastGain: number;
}

export const initialGame: GameState = { points: 0, combo: 0, bestCombo: 0, cleanReps: 0, lastGain: 0 };

export const BASE_POINTS = 10;
export const COMBO_BONUS = 2;
export const COMBO_CAP = 10;
export const EFFORT_POINTS = 2;

/** Correct reps earn 10 + 2 x current combo (capped); other reps earn 2 and reset the combo. */
export function gameReducer(state: GameState, label: RepOut["label"]): GameState {
  if (label === "correct") {
    const gain = BASE_POINTS + COMBO_BONUS * Math.min(state.combo, COMBO_CAP);
    const combo = state.combo + 1;
    return {
      points: state.points + gain,
      combo,
      bestCombo: Math.max(state.bestCombo, combo),
      cleanReps: state.cleanReps + 1,
      lastGain: gain,
    };
  }
  return { ...state, points: state.points + EFFORT_POINTS, combo: 0, lastGain: EFFORT_POINTS };
}
