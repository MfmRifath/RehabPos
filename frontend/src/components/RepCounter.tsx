/**
 * Live rep count, sized to read from across the room. Only the count itself is a live region:
 * the points beside it and the angle gauge change too often to announce.
 */
export function RepCounter({ count, points, combo, gain, gainKey }: { count: number; points: number; combo: number; gain: number; gainKey: number }) {
  return (
    <div className="rep-counter">
      <div className="rep-number" data-testid="rep-count" aria-live="polite" aria-atomic="true">
        <span key={count} className="rep-pop">{count}</span>
      </div>
      <div className="rep-label">reps</div>
      <div className="rep-points">{points} pts · combo {combo}</div>
      {gain > 0 && <span key={gainKey} className="rep-gain" aria-hidden="true">+{gain}</span>}
    </div>
  );
}
