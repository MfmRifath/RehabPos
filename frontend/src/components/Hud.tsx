export interface HudStats {
  fps: number;
  inferMs: number;
  rttMs: number | null;
  serverMs: number | null;
  dropped: number;
  delegate: string;
  socket: string;
}

const f = (v: number | null, d = 1) => (v === null ? "–" : v.toFixed(d));

export function Hud({ s }: { s: HudStats }) {
  return (
    <div className="hud" data-testid="hud">
      <div>FPS {f(s.fps)}</div>
      <div>pose {f(s.inferMs)} ms ({s.delegate})</div>
      <div>rtt {f(s.rttMs)} ms</div>
      <div>server {f(s.serverMs, 2)} ms</div>
      <div>dropped {s.dropped}</div>
      <div>ws {s.socket}</div>
    </div>
  );
}
