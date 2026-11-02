import type { ReactNode } from "react";

const HEAD_SIDE = <circle cx="30" cy="11" r="6" />;
const HEAD_FRONT = <circle cx="32" cy="11" r="6" />;
const BODY_SIDE = <path d="M30 21 V42 M30 42 L26 60 M30 42 L34 60" />;
const BODY_FRONT = <path d="M22 22 H42 M32 22 V42 M32 42 L26 60 M32 42 L38 60 M22 22 L19 42" />;

// Each figure draws the resting body, then the moving limb and its path in the accent colour.
const FIGURES: Record<string, ReactNode> = {
  shoulder_flexion: (
    <>
      {HEAD_SIDE}
      {BODY_SIDE}
      <path className="moving" d="M30 22 L46 8" />
      <path className="motion" d="M30 43 A 21 21 0 0 0 46 8" strokeDasharray="1 5" />
    </>
  ),
  shoulder_abduction: (
    <>
      {HEAD_FRONT}
      {BODY_FRONT}
      <path className="moving" d="M42 22 L58 12" />
      <path className="motion" d="M44 42 A 20 20 0 0 0 58 12" strokeDasharray="1 5" />
    </>
  ),
  elbow_flexion: (
    <>
      {HEAD_SIDE}
      {BODY_SIDE}
      <path d="M30 22 L30 36" />
      <path className="moving" d="M30 36 L43 26" />
      <path className="motion" d="M43 44 A 15 15 0 0 0 43 26" strokeDasharray="1 5" />
    </>
  ),
  forward_reach: (
    <>
      {HEAD_SIDE}
      {BODY_SIDE}
      <path className="moving" d="M30 23 H48" />
      <path className="motion" d="M52 23 H60 M56 19 L60 23 L56 27" />
    </>
  ),
  shoulder_external_rotation: (
    <>
      {HEAD_FRONT}
      {BODY_FRONT}
      <path d="M42 22 L44 36" />
      <path className="moving" d="M44 36 L58 32" />
      <path className="motion" d="M46 50 A 14 14 0 0 0 58 32" strokeDasharray="1 5" />
    </>
  ),
};

const FALLBACK = (
  <>
    {HEAD_FRONT}
    {BODY_FRONT}
    <path d="M42 22 L45 42" />
  </>
);

/** Decorative stick figure showing how the exercise moves. */
export function ExercisePictogram({ id }: { id: string }) {
  return (
    <svg className="pictogram" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {FIGURES[id] ?? FALLBACK}
    </svg>
  );
}
