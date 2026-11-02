import { CircleAlert, EyeOff, Gauge, LoaderCircle, Pause, Play, Square, Volume2, VolumeX, WifiOff } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { CueBanner } from "../components/CueBanner";
import { Hud, type HudStats } from "../components/Hud";
import { RepCounter } from "../components/RepCounter";
import { RomGauge } from "../components/RomGauge";
import { api } from "../lib/api";
import { describeApiError, describeCameraError } from "../lib/errors";
import { formatMeasure, REP_LABELS } from "../lib/format";
import { CueSpeaker } from "../lib/speech";
import { drawOverlay, OVERLAY_COLORS } from "../pose/draw";
import { useCameraPose } from "../pose/useCameraPose";
import { usePoseLoop, type FrameResult } from "../pose/usePoseLoop";
import { navigate } from "../router";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";
import { SessionSocket, sessionUrl, type SocketStatus } from "../ws/client";
import { landmarksMessage, type Phase } from "../ws/protocol";

const PHASES: { id: Phase; label: string }[] = [
  { id: "rest", label: "Rest" },
  { id: "ascending", label: "Moving" },
  { id: "peak", label: "Peak" },
  { id: "descending", label: "Returning" },
];

const SOCKET_TEXT: Partial<Record<SocketStatus, string>> = {
  connecting: "Connecting…",
  reconnecting: "Connection lost. Reconnecting…",
  closed: "Disconnected. Reps aren't being counted.",
};

export function Live() {
  const sel = useApp((s) => s.selection);
  const userId = useApp((s) => s.profile.userId);
  const live = useApp((s) => s.live);
  const muted = useApp((s) => s.muted);
  const showHud = useApp((s) => s.showHud);
  const { exercises, error: exercisesError, reload } = useExercises();
  const ex = exercises.find((e) => e.id === sel.exerciseId);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<SessionSocket | null>(null);
  const speaker = useRef(new CueSpeaker());
  const frameIdx = useRef(0);
  const { videoRef, video, landmarker, delegate, startup, error: cameraError, retry: retryCamera } = useCameraPose(sel.model);
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("connecting");
  const [status, setStatus] = useState("Creating session…");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [sessionAttempt, setSessionAttempt] = useState(0);
  const [running, setRunning] = useState(true);
  const [personInView, setPersonInView] = useState(true);
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
      .catch((err: unknown) => {
        if (!cancelled) setSessionError(describeApiError(err));
      });
    return () => {
      cancelled = true;
      socket?.close();
      socketRef.current = null;
    };
  }, [ex, userId, sel.side, sel.view, sessionAttempt]);

  // 2) per-frame: send landmarks, draw overlay
  const onFrame = useCallback(
    (r: FrameResult) => {
      const lm = r.result.landmarks[0] ?? null;
      const world = r.result.worldLandmarks[0] ?? null;
      setPersonInView(lm !== null);
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
          colors: OVERLAY_COLORS,
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
    // Keep the button in step with the server: nothing changes if the message could not be sent.
    if (socketRef.current?.sendControl(next ? "start_set" : "pause")) setRunning(next);
  }, [running]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === " ") {
        if (tag === "BUTTON" || tag === "A") return; // Space already activates the focused control.
        e.preventDefault();
        if (!e.repeat) toggleRun();
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

  const hud: HudStats = { fps: stats.fps, inferMs: stats.inferMs, rttMs: live.rttMs, serverMs: live.serverMs, dropped: live.dropped, delegate: delegate ?? "–", socket: socketStatus };
  const primary = ex ? (live.angles[ex.primary_angle] ?? null) : null;
  const lastRep = live.reps.at(-1);
  const camera = startup === "error" ? describeCameraError(cameraError) : null;
  const socketText = SOCKET_TEXT[socketStatus];
  const blocked = camera ?? (sessionError ? { title: "The session didn't start", detail: sessionError } : exercisesError ? { title: "Exercises didn't load", detail: exercisesError } : null);
  const retryBlocked = camera
    ? retryCamera
    : sessionError
      ? () => {
          setSessionError(null);
          setSessionAttempt((a) => a + 1);
        }
      : reload;

  return (
    <main className="live" tabIndex={-1}>
      <div className="live-stage">
        <video ref={videoRef} className="mirror" />
        <canvas ref={canvasRef} />
        {blocked ? (
          <div className="stage-overlay" role="alert">
            <CircleAlert size={44} aria-hidden="true" />
            <h2>{blocked.title}</h2>
            <p>{blocked.detail}</p>
            <div className="btn-row">
              <button className="btn" onClick={retryBlocked}>Try again</button>
              <a className="btn btn-ghost" href="#/">Back to exercises</a>
            </div>
          </div>
        ) : startup !== "ready" ? (
          <div className="stage-overlay" role="status">
            <LoaderCircle size={44} className="spin" aria-hidden="true" />
            <p>{startup === "camera" ? "Starting camera…" : "Loading pose model…"}</p>
          </div>
        ) : null}
      </div>
      {!running && !blocked && (
        <div className="paused" role="status">
          <strong>Paused</strong>
          <span>Press space to continue</span>
        </div>
      )}
      <div className="live-ui">
        <header className="live-top">
          <div>
            <RepCounter count={live.repCount} points={live.game.points} combo={live.game.combo} gain={live.game.lastGain} gainKey={live.reps.length} />
            {lastRep && ex && (
              <p className="rep-last">
                Last: {formatMeasure(lastRep.peak_angle, ex.unit)}, {lastRep.duration_s.toFixed(1)} s, {REP_LABELS[lastRep.label]}
              </p>
            )}
          </div>
          <div className="live-title">
            <h1>{ex?.name ?? "…"}</h1>
            <p>{sel.side === "R" ? "Right" : "Left"} arm</p>
            <p className="live-status" data-testid="status">{status}</p>
          </div>
          <div className="live-right">
            {ex && <RomGauge value={primary} target={ex.target} unit={ex.unit} />}
            {showHud && <Hud s={hud} />}
          </div>
        </header>
        <div className="live-pills" aria-live="polite">
          {socketText && !blocked && (
            <span className="pill pill-alert"><WifiOff size={20} aria-hidden="true" />{socketText}</span>
          )}
          {live.error && <span className="pill pill-alert"><CircleAlert size={20} aria-hidden="true" />{live.error}</span>}
          {landmarker && !personInView && running && (
            <span className="pill pill-alert"><EyeOff size={20} aria-hidden="true" />Can't see you. Step into the camera's view.</span>
          )}
          {!live.calibrated && live.sid && socketStatus === "open" && <span className="pill">Calibrating: hold still for a moment</span>}
          {live.hints.map((h) => (
            <span key={h} className="pill">{h}</span>
          ))}
        </div>
        <div className="live-bottom">
          <CueBanner cue={live.cue} />
          <ol className="phase-strip" aria-label="Movement phase">
            {PHASES.map((p) => (
              <li key={p.id} aria-current={live.phase === p.id ? "step" : undefined}>{p.label}</li>
            ))}
          </ol>
          <div className="live-controls">
            <div className="live-controls-group">
              <button className="ctrl" onClick={toggleRun} data-testid="toggle-run">
                {running ? <Pause size={20} aria-hidden="true" /> : <Play size={20} aria-hidden="true" />}
                <span className="ctrl-text">{running ? "Pause" : "Start set"}</span>{" "}
                <kbd>(space)</kbd>
              </button>
              <button className="ctrl" onClick={() => useApp.getState().toggleMute()}>
                {muted ? <VolumeX size={20} aria-hidden="true" /> : <Volume2 size={20} aria-hidden="true" />}
                <span className="ctrl-text">{muted ? "Unmute" : "Mute"}</span>{" "}
                <kbd>(m)</kbd>
              </button>
              <button className="ctrl" onClick={() => useApp.getState().toggleHud()} aria-pressed={showHud}>
                <Gauge size={20} aria-hidden="true" />
                <span className="ctrl-text">Stats</span>{" "}
                <kbd>(h)</kbd>
              </button>
            </div>
            <button className="ctrl ctrl-end" onClick={end} data-testid="end">
              <Square size={18} aria-hidden="true" />
              <span className="ctrl-text">End session</span>
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
