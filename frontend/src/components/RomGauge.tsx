import { Check } from "lucide-react";
import { formatMeasure, targetFraction } from "../lib/format";

const R = 50;
const SWEEP = 270; // degrees of arc; the gap sits at the bottom
const LENGTH = 100; // pathLength, so the dash array reads as a percentage

function arcPath(): string {
  const point = (deg: number) => {
    const rad = (deg * Math.PI) / 180;
    return `${60 + R * Math.cos(rad)} ${60 + R * Math.sin(rad)}`;
  };
  return `M ${point(135)} A ${R} ${R} 0 1 1 ${point(135 + SWEEP)}`;
}

/**
 * Current range of motion against the target, in degrees or as a ratio. The arc stops filling at
 * the target and shows a tick: the gauge never rewards going further (docs/09 §3).
 */
export function RomGauge({ value, target, unit }: { value: number | null; target: number; unit: string }) {
  const fraction = targetFraction(value, target);
  const shown = formatMeasure(value, unit);
  const goal = formatMeasure(target, unit);
  return (
    <div className="gauge" role="img" aria-label={`Current ${shown}, target ${goal}`} data-testid="rom-gauge">
      <svg viewBox="0 0 120 120" fill="none" aria-hidden="true">
        <path className="gauge-track" d={arcPath()} pathLength={LENGTH} strokeWidth="9" strokeLinecap="round" />
        <path
          className="gauge-fill"
          d={arcPath()}
          pathLength={LENGTH}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${(fraction * LENGTH).toFixed(1)} ${LENGTH}`}
          visibility={fraction > 0 ? "visible" : "hidden"}
          data-testid="rom-gauge-fill"
        />
      </svg>
      <div className="gauge-text">
        <span className="gauge-value">{shown}</span>
        <span className={`gauge-target ${fraction >= 1 ? "gauge-reached" : ""}`}>
          {fraction >= 1 && <Check size={14} strokeWidth={3} aria-hidden="true" />}
          target {goal}
        </span>
      </div>
    </div>
  );
}
