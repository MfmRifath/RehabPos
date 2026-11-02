// Small shared UI pieces: state panels, skeletons, stat tiles, stepper, segmented control, brand mark.
import { Check, CircleAlert, Inbox, LoaderCircle } from "lucide-react";
import type { ReactNode } from "react";

export function BrandMark({ size = 28 }: { size?: number }) {
  // The joint-and-arc motif: the same arc the live overlay and the range gauge draw.
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path d="M6 25 A 19 19 0 0 1 25 6" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M8 24 L17 15" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" opacity="0.45" />
      <circle cx="7" cy="25" r="3.5" fill="currentColor" />
      <circle cx="19" cy="13" r="3.5" fill="currentColor" />
    </svg>
  );
}

type StateTone = "loading" | "empty" | "error";

const STATE_ICON = { loading: LoaderCircle, empty: Inbox, error: CircleAlert } as const;

/** Loading, empty and error messages, with an optional next action. */
export function StatePanel({ tone, title, children, action }: { tone: StateTone; title: string; children?: ReactNode; action?: ReactNode }) {
  const Icon = STATE_ICON[tone];
  return (
    <div className={`card state-panel state-${tone}`} role={tone === "error" ? "alert" : "status"}>
      <span className="state-icon">
        <Icon size={26} aria-hidden="true" className={tone === "loading" ? "spin" : undefined} />
      </span>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Skeleton({ className }: { className: string }) {
  return <span className={`skeleton ${className}`} aria-hidden="true" />;
}

export function StatTile({ label, value, note, icon }: { label: string; value: ReactNode; note?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="stat-label">
        {icon}
        {label}
      </div>
      <div className="stat-value">{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

const STEPS = ["Choose", "Camera check", "Exercise", "Review"];

/** Where the person is in the session flow; `current` is a 0-based index into the four steps. */
export function Stepper({ current }: { current: number }) {
  return (
    <ol className="stepper" aria-label="Session steps">
      {STEPS.map((name, i) => {
        const state = i < current ? "step-done" : i === current ? "step-current" : "";
        return (
          <li key={name} className={`step ${state}`} aria-current={i === current ? "step" : undefined}>
            <span className="step-dot" aria-hidden="true">
              {i < current ? <Check size={15} strokeWidth={3} /> : i + 1}
            </span>
            <span className="step-name">{name}</span>
          </li>
        );
      })}
    </ol>
  );
}

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

/** A radio group drawn as a segmented control; arrow keys come from the native radios. */
export function Segmented<T extends string>({
  legend,
  name,
  value,
  options,
  onChange,
  testId,
}: {
  legend: string;
  name: string;
  value: T;
  options: SegmentedOption<T>[];
  onChange: (value: T) => void;
  testId?: string;
}) {
  return (
    <fieldset className="segmented" data-testid={testId}>
      <legend>{legend}</legend>
      <div className="segmented-track">
        {options.map((o) => (
          <label key={o.value} className="segmented-option">
            <input type="radio" name={name} value={o.value} checked={o.value === value} onChange={() => onChange(o.value)} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
