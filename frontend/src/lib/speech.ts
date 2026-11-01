// Spoken cues: the server already rate-limits `speak`; this adds a client-side guard so a
// reconnect or duplicate message can never speak more often than once per interval.
import type { CueOut } from "../ws/protocol";

export class CueSpeaker {
  private lastAt = -Infinity;
  constructor(
    private readonly intervalMs = 3000,
    private readonly say: (text: string) => void = speakWithBrowser,
  ) {}

  /** Returns true if the cue was spoken. */
  handle(cue: CueOut | null, nowMs: number, muted: boolean): boolean {
    if (!cue || !cue.speak || muted) return false;
    if (nowMs - this.lastAt < this.intervalMs) return false;
    this.lastAt = nowMs;
    this.say(cue.text);
    return true;
  }
}

export function speakWithBrowser(text: string): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  window.speechSynthesis.speak(u);
}
