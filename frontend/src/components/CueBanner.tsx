import type { CueOut } from "../ws/protocol";

const LABEL = { info: "Tip", warn: "Check form", error: "Setup" } as const;
const ICON = { info: "ℹ", warn: "⚠", error: "⛔" } as const;

/** Single cue banner; colour by severity plus icon + label (never colour alone). Captions TTS. */
export function CueBanner({ cue }: { cue: CueOut | null }) {
  if (!cue) return null;
  const sev = cue.severity ?? "info";
  return (
    <div className={`cue ${sev}`} role="status" aria-live="polite" data-testid="cue">
      <span aria-hidden="true">{ICON[sev]}</span>
      <span className="sr-only">{LABEL[sev]}:</span>
      <span>{cue.text}</span>
    </div>
  );
}
