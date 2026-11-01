import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";
import { CueBanner } from "../components/CueBanner";
import { Hud, type HudStats } from "../components/Hud";
import { api } from "../lib/api";
import { CueSpeaker } from "../lib/speech";
import { startCamera, stopCamera } from "../pose/camera";
import { drawOverlay } from "../pose/draw";
import { getLandmarker, type Delegate } from "../pose/landmarker";
import { usePoseLoop, type FrameResult } from "../pose/usePoseLoop";
import { navigate } from "../router";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";
import { SessionSocket, sessionUrl, type SocketStatus } from "../ws/client";
import { landmarksMessage, type Phase } from "../ws/protocol";

const PHASES: Phase[] = ["rest", "ascending", "peak", "descending"];

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function Live() {
  const sel = useApp((s) => s.selection);
  const userId = useApp((s) => s.profile.userId);
  const live = useApp((s) => s.live);
  const muted = useApp((s) => s.muted);
  const showHud = useApp((s) => s.showHud);
  const { exercises } = useExercises();
  const ex = exercises.find((e) => e.id === sel.exerciseId);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<SessionSocket | null>(null);
  const speaker = useRef(new CueSpeaker());
  const frameIdx = useRef(0);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [landmarker, setLandmarker] = useState<PoseLandmarker | null>(null);
  const [delegate, setDelegate] = useState<Delegate | "–">("–");
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("connecting");
  const [status, setStatus] = useState("Creating session…");
  const [running, setRunning] = useState(true);
  const [stats, setStats] = useState({ fps: 0, inferMs: 0 });
  const lastStatsAt = useRef(0);

  // 1) session + socket
  useEffect(() => {
    if (!ex) return;
    let socket: SessionSocket | null = null;
    let cancelled = false;
    api
      .createSession({ user_id: userId, exercise_id: ex.id, side: sel.side, view: sel.view ?? ex.view })
      .then((s) => {
        if (cancelled) return;
        useApp.getState().resetLive(s.id);
        socket = new SessionSocket({
          url: sessionUrl(s.id),
          onStatus: setSocketStatus,
          onMessage: (msg, rtt) => {
            useApp.getState().applyMessage(msg, rtt);
            if (msg.type === "feedback") {
              speaker.current.handle(msg.cue, performance.now(), useApp.getState().muted);
            }
          },
        });
        socketRef.current = socket;
        setStatus(`Session ${s.id}`);
      })
      .catch((err: unknown) => setStatus(`Cannot create session: ${String(err)}`));
    return () => {
      cancelled = true;
      socket?.close();
      socketRef.current = null;
    };
  }, [ex, userId, sel.side, sel.view]);

  // 2) camera + pose model
  useEffect(() => {
    let stream: MediaStream | null = null;
    let lm: PoseLandmarker | null;
    let cancelled = false;
    (async () => {
      try {
        const v = videoRef.current as HTMLVideoElement;
        stream = await startCamera(v);
        setVideo(v);
        const created = await getLandmarker(sel.model);
        if (cancelled) return;
        lm = created.landmarker;
        setLandmarker(lm);
        setDelegate(created.delegate);
      } catch (err) {
        setStatus(`Camera or model error: ${String(err)}`);
      }
    })();
    return () => {
      cancelled = true;
      stopCamera(stream);
    };
  }, [sel.model]);

  // 3) per-frame: send landmarks, draw overlay
  const onFrame = useCallback(
    (r: FrameResult) => {
      const lm = r.result.landmarks[0] ?? null;
      const world = r.result.worldLandmarks[0] ?? null;
      const sid = useApp.getState().live.sid;
      if (lm && world && sid && socketRef.current) {
        socketRef.current.sendLandmarks(
          landmarksMessage({ sid, frame: frameIdx.current++, tsMs: r.tsMs, width: r.width, height: r.height, landmarks: lm, world, inferMs: r.inferMs }),
        );
      }
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx && ex) {
        if (canvas.width !== r.width) canvas.width = r.width;
        if (canvas.height !== r.height) canvas.height = r.height;
        drawOverlay(ctx, lm, r.width, r.height, {
          side: sel.side,
          primary: ex.primary_angle,
          angle: useApp.getState().live.angles[ex.primary_angle] ?? null,
          target: ex.target,
          unit: ex.unit,
          mirror: true,
          colors: { bone: "rgba(255,255,255,0.85)", active: cssVar("--warning", "#fab219"), band: "rgba(12,163,12,0.55)", text: "#fff" },
        });
      }
      const now = performance.now();
      if (now - lastStatsAt.current > 500) {
        lastStatsAt.current = now;
        setStats({ fps: r.fps, inferMs: r.inferMs });
      }
    },
    [ex, sel.side],
  );
  usePoseLoop(video, landmarker, onFrame);

  const toggleRun = useCallback(() => {
    const next = !running;
    socketRef.current?.sendControl(next ? "start_set" : "pause");
    setRunning(next);
  }, [running]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement | null)?.tagName === "INPUT") return;
      if (e.key === " ") {
        e.preventDefault();
        toggleRun();
      } else if (e.key === "m") useApp.getState().toggleMute();
      else if (e.key === "h") useApp.getState().toggleHud();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleRun]);

  const end = async () => {
    const sid = live.sid;
    socketRef.current?.close();
    if (sid) {
      await api.endSession(sid).catch(() => undefined);
      navigate(`/summary/${sid}`);
    } else navigate("/");
  };

  const hud: HudStats = { fps: stats.fps, inferMs: stats.inferMs, rttMs: live.rttMs, serverMs: live.serverMs, dropped: live.dropped, delegate, socket: socketStatus };
  const unit = ex?.unit === "deg" ? "°" : "";
  const primary = ex ? live.angles[ex.primary_angle] : null;

  return (
    <section>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1 style={{ margin: "8px 0" }}>
          {ex?.name ?? "…"} <span className="muted">({sel.side === "R" ? "right" : "left"} arm)</span>
        </h1>
        <div className="row">
          <button onClick={toggleRun} data-testid="toggle-run">{running ? "Pause (space)" : "Start set (space)"}</button>
          <button onClick={() => useApp.getState().toggleMute()} aria-pressed={muted}>{muted ? "Unmute (m)" : "Mute (m)"}</button>
          <button onClick={() => useApp.getState().toggleHud()} aria-pressed={showHud}>HUD (h)</button>
          <button className="primary" onClick={end} data-testid="end">End session</button>
        </div>
      </div>
      <p className="muted" data-testid="status">{status}{live.error ? ` · ${live.error}` : ""}</p>
      <div className="stage">
        <video ref={videoRef} className="mirror" />
        <canvas ref={canvasRef} />
        <div className="counter" aria-live="polite">
          <div className="big" data-testid="rep-count">{live.repCount}</div>
          <div>reps · {primary != null ? `${Math.round(primary)}${unit}` : "–"} / target {ex?.target ?? "–"}{unit}</div>
          <div>{live.game.points} pts · combo {live.game.combo}</div>
        </div>
        {showHud && <Hud s={hud} />}
        <CueBanner cue={live.cue} />
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        <div className="phase" aria-label="Movement phase">
          {PHASES.map((p) => (
            <span key={p} className={live.phase === p ? "on" : ""}>{p}</span>
          ))}
        </div>
        {!live.calibrated && <span className="tag">calibrating: hold still for a moment</span>}
        {live.hints.map((h) => (
          <span key={h} className="tag">{h}</span>
        ))}
      </div>
      {live.reps.length > 0 && (
        <p className="muted">
          Last rep: {live.reps.at(-1)?.peak_angle.toFixed(0)}{unit}, {live.reps.at(-1)?.duration_s.toFixed(1)} s,{" "}
          {live.reps.at(-1)?.label.replace("_", " ")} {live.game.lastGain > 0 && `(+${live.game.lastGain} pts)`}
        </p>
      )}
    </section>
  );
}
