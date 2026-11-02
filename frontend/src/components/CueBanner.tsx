import { Camera, Lightbulb, TriangleAlert } from "lucide-react";
import type { CueOut } from "../ws/protocol";

const LABEL = { info: "Tip", warn: "Check form", error: "Setup" } as const;
const ICON = { info: Lightbulb, warn: TriangleAlert, error: Camera } as const;

/**
 * Single cue banner; colour by severity plus icon + label (never colour alone). Captions TTS.
 * The live region stays mounted so screen readers announce a cue the moment it appears.
 */
export function CueBanner({ cue }: { cue: CueOut | null }) {
  const sev = cue?.severity ?? "info";
  const Icon = ICON[sev];
  return (
    <div role="status" aria-live="polite">
      {cue && (
        <div className={`cue ${sev}`} data-testid="cue">
          <Icon aria-hidden="true" />
          <span>
            <span className="cue-label">{LABEL[sev]}:</span>
            <span className="cue-text">{cue.text}</span>
          </span>
        </div>
      )}
    </div>
  );
}
