// Reconnecting WebSocket client for /ws/session/{sid} with round-trip measurement.
import { controlMessage, parseServerMessage, type ControlAction, type LandmarksMsg, type ServerMessage } from "./protocol";

export type SocketStatus = "connecting" | "open" | "reconnecting" | "closed";

export interface SocketOptions {
  url: string;
  onMessage: (msg: ServerMessage, rttMs: number | null) => void;
  onStatus?: (status: SocketStatus) => void;
  maxRetries?: number;
  /** Factory for tests. */
  createSocket?: (url: string) => WebSocket;
  now?: () => number;
}

/** Default WS URL for a session, same origin as the page (works behind the Vite proxy). */
export function sessionUrl(sid: string, loc: Location = window.location): string {
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${loc.host}/ws/session/${encodeURIComponent(sid)}`;
}

export class SessionSocket {
  private ws: WebSocket | null = null;
  private retries = 0;
  private stopped = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private readonly sentAt = new Map<number, number>();
  private readonly opts: Required<Omit<SocketOptions, "onStatus">> & Pick<SocketOptions, "onStatus">;
  status: SocketStatus = "connecting";
  lastRttMs: number | null = null;

  constructor(opts: SocketOptions) {
    this.opts = {
      maxRetries: 8,
      createSocket: (u) => new WebSocket(u),
      now: () => performance.now(),
      ...opts,
    };
    this.connect();
  }

  private setStatus(s: SocketStatus): void {
    this.status = s;
    this.opts.onStatus?.(s);
  }

  private connect(): void {
    const ws = this.opts.createSocket(this.opts.url);
    this.ws = ws;
    ws.onopen = () => {
      this.retries = 0;
      this.setStatus("open");
    };
    ws.onmessage = (ev: MessageEvent) => {
      const msg = parseServerMessage(String(ev.data));
      if (!msg) return;
      let rtt: number | null = null;
      if (msg.type === "feedback") {
        const t0 = this.sentAt.get(msg.frame);
        if (t0 !== undefined) {
          rtt = this.opts.now() - t0;
          this.lastRttMs = rtt;
        }
        for (const k of this.sentAt.keys()) if (k <= msg.frame) this.sentAt.delete(k);
      }
      this.opts.onMessage(msg, rtt);
    };
    ws.onclose = (ev: CloseEvent) => {
      if (this.stopped || ev.code === 4404) {
        this.setStatus("closed");
        return;
      }
      if (this.retries >= this.opts.maxRetries) {
        this.setStatus("closed");
        return;
      }
      const delay = Math.min(250 * 2 ** this.retries, 5000);
      this.retries += 1;
      this.setStatus("reconnecting");
      this.timer = setTimeout(() => this.connect(), delay);
    };
  }

  /** Send one frame; silently skipped while not connected (live frames are disposable). */
  sendLandmarks(msg: LandmarksMsg): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.sentAt.set(msg.frame, this.opts.now());
    if (this.sentAt.size > 300) {
      const oldest = this.sentAt.keys().next().value;
      if (oldest !== undefined) this.sentAt.delete(oldest);
    }
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  sendControl(action: ControlAction): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(controlMessage(action)));
    return true;
  }

  close(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
    this.setStatus("closed");
  }
}
