import { SessionSocket, sessionUrl } from "./client";
import { landmarksMessage } from "./protocol";

class FakeSocket {
  static instances: FakeSocket[] = [];
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }
  send(s: string) {
    this.sent.push(s);
  }
  close() {
    this.readyState = 3;
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
}

const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 1 }));
const frame = (n: number) => landmarksMessage({ sid: "s", frame: n, tsMs: n, width: 2, height: 2, landmarks: lm, world: lm });
const fb = (n: number) => JSON.stringify({ v: 1, type: "feedback", frame: n, ts_ms: n, angles: {}, phase: "rest", rep: { count: 0 }, flags: [], cue: null, quality: { visibility_ok: true, view_ok: true }, server_ms: 0.1 });

beforeEach(() => {
  FakeSocket.instances = [];
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

function make(now: { t: number }, onMessage = vi.fn(), onStatus = vi.fn()) {
  const s = new SessionSocket({
    url: "ws://x/ws/session/s",
    onMessage,
    onStatus,
    createSocket: (u) => new FakeSocket(u) as unknown as WebSocket,
    now: () => now.t,
  });
  return { s, onMessage, onStatus, sock: () => FakeSocket.instances.at(-1) as FakeSocket };
}

it("sends only while open and measures round-trip per frame", () => {
  const now = { t: 0 };
  const { s, onMessage, sock } = make(now);
  expect(s.sendLandmarks(frame(1))).toBe(false);
  sock().open();
  expect(s.sendLandmarks(frame(1))).toBe(true);
  now.t = 25;
  sock().onmessage?.({ data: fb(1) });
  expect(onMessage).toHaveBeenLastCalledWith(expect.objectContaining({ type: "feedback", frame: 1 }), 25);
  expect(s.lastRttMs).toBe(25);
  sock().onmessage?.({ data: "garbage" });
  expect(onMessage).toHaveBeenCalledTimes(1);
  expect(s.sendControl("pause")).toBe(true);
  expect(JSON.parse(sock().sent.at(-1) as string)).toEqual({ v: 1, type: "control", action: "pause" });
});

it("reconnects with backoff and stops on 4404 or close()", () => {
  const { s, onStatus, sock } = make({ t: 0 });
  sock().open();
  sock().onclose?.({ code: 1006 });
  expect(onStatus).toHaveBeenLastCalledWith("reconnecting");
  vi.advanceTimersByTime(250);
  expect(FakeSocket.instances).toHaveLength(2);
  sock().onclose?.({ code: 4404 });
  expect(onStatus).toHaveBeenLastCalledWith("closed");
  vi.advanceTimersByTime(10_000);
  expect(FakeSocket.instances).toHaveLength(2);
  s.close();
  expect(s.status).toBe("closed");
});

it("gives up after maxRetries", () => {
  const onStatus = vi.fn();
  new SessionSocket({ url: "u", onMessage: vi.fn(), onStatus, maxRetries: 2, createSocket: (u) => new FakeSocket(u) as unknown as WebSocket });
  for (let i = 0; i < 3; i++) {
    (FakeSocket.instances.at(-1) as FakeSocket).onclose?.({ code: 1006 });
    vi.advanceTimersByTime(5000);
  }
  expect(onStatus).toHaveBeenLastCalledWith("closed");
  expect(FakeSocket.instances).toHaveLength(3);
});

it("builds same-origin ws/wss URLs", () => {
  expect(sessionUrl("a b", { protocol: "http:", host: "h:1" } as Location)).toBe("ws://h:1/ws/session/a%20b");
  expect(sessionUrl("s", { protocol: "https:", host: "h" } as Location)).toBe("wss://h/ws/session/s");
});
