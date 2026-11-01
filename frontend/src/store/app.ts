import { create } from "zustand";
import { gameReducer, initialGame, type GameState } from "../lib/gamification";
import type { View } from "../lib/setup";
import type { CueOut, FlagOut, Phase, RepOut, ServerMessage } from "../ws/protocol";

export type ModelVariant = "lite" | "full" | "heavy";
export type Side = "L" | "R";

export interface Selection {
  exerciseId: string;
  side: Side;
  view: View | null;
  model: ModelVariant;
}

export interface LiveState {
  sid: string | null;
  phase: Phase;
  repCount: number;
  angles: Record<string, number | null>;
  flags: FlagOut[];
  cue: CueOut | null;
  visibilityOk: boolean;
  viewOk: boolean;
  calibrated: boolean;
  hints: string[];
  reps: RepOut[];
  serverMs: number | null;
  rttMs: number | null;
  dropped: number;
  error: string | null;
  game: GameState;
}

export const initialLive: LiveState = {
  sid: null,
  phase: "rest",
  repCount: 0,
  angles: {},
  flags: [],
  cue: null,
  visibilityOk: false,
  viewOk: true,
  calibrated: false,
  hints: [],
  reps: [],
  serverMs: null,
  rttMs: null,
  dropped: 0,
  error: null,
  game: initialGame,
};

/** Pure reducer: apply one server message to the live view state. */
export function liveReducer(s: LiveState, msg: ServerMessage, rttMs: number | null = null): LiveState {
  switch (msg.type) {
    case "feedback":
      return {
        ...s,
        phase: msg.phase,
        repCount: msg.rep.count,
        angles: msg.angles,
        flags: msg.flags,
        cue: msg.cue,
        visibilityOk: msg.quality.visibility_ok,
        viewOk: msg.quality.view_ok,
        serverMs: msg.server_ms,
        rttMs: rttMs ?? s.rttMs,
        dropped: msg.dropped ?? s.dropped,
        error: null,
      };
    case "rep_completed":
      if (s.reps.some((r) => r.idx === msg.rep.idx)) return s; // duplicate after reconnect
      return { ...s, reps: [...s.reps, msg.rep], game: gameReducer(s.game, msg.rep.label) };
    case "setup_status":
      return { ...s, calibrated: msg.calibrated, visibilityOk: msg.visibility_ok, viewOk: msg.view_ok, hints: msg.hints };
    case "error":
      return { ...s, error: `${msg.code}: ${msg.message}` };
  }
}

const KEY = "rehabpose.profile.v1";

interface Profile {
  userId: string;
  disclaimerAccepted: boolean;
}

function loadProfile(): Profile {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as Profile;
  } catch {
    /* storage unavailable: fall through */
  }
  const id = `u-${Math.random().toString(36).slice(2, 10)}`;
  return { userId: id, disclaimerAccepted: false };
}

function saveProfile(p: Profile): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* ignore */
  }
}

interface AppStore {
  profile: Profile;
  selection: Selection;
  live: LiveState;
  muted: boolean;
  showHud: boolean;
  acceptDisclaimer: () => void;
  setUserId: (id: string) => void;
  select: (patch: Partial<Selection>) => void;
  resetLive: (sid: string | null) => void;
  applyMessage: (msg: ServerMessage, rtt: number | null) => void;
  toggleMute: () => void;
  toggleHud: () => void;
}

export const useApp = create<AppStore>((set) => ({
  profile: loadProfile(),
  selection: { exerciseId: "shoulder_flexion", side: "R", view: null, model: "full" },
  live: initialLive,
  muted: false,
  showHud: false,
  acceptDisclaimer: () =>
    set((s) => {
      const profile = { ...s.profile, disclaimerAccepted: true };
      saveProfile(profile);
      return { profile };
    }),
  setUserId: (userId) =>
    set((s) => {
      const profile = { ...s.profile, userId };
      saveProfile(profile);
      return { profile };
    }),
  select: (patch) => set((s) => ({ selection: { ...s.selection, ...patch } })),
  resetLive: (sid) => set({ live: { ...initialLive, sid } }),
  applyMessage: (msg, rtt) => set((s) => ({ live: liveReducer(s.live, msg, rtt) })),
  toggleMute: () => set((s) => ({ muted: !s.muted })),
  toggleHud: () => set((s) => ({ showHud: !s.showHud })),
}));
