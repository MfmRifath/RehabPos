import { ArrowRight, CircleAlert, CircleCheck, CircleDashed, HeartPulse, LoaderCircle } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { Stepper } from "../components/ui";
import { describeCameraError } from "../lib/errors";
import { setupChecklist, setupReport, type CheckState, type SetupReport } from "../lib/setup";
import { drawOverlay, OVERLAY_COLORS } from "../pose/draw";
import { useCameraPose } from "../pose/useCameraPose";
import { usePoseLoop, type FrameResult } from "../pose/usePoseLoop";
import { navigate } from "../router";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";

const CHECK_ICON = { pending: CircleDashed, ok: CircleCheck, problem: CircleAlert } as const;
// Spoken state: the icon and background alone say nothing to a screen reader.
const CHECK_WORD: Record<CheckState, string> = { pending: "Waiting:", ok: "Done:", problem: "Needs attention:" };

export function Setup() {
  const sel = useApp((s) => s.selection);
  const { exercises } = useExercises();
  const ex = exercises.find((e) => e.id === sel.exerciseId);
  const view = sel.view ?? ex?.view ?? "sagittal";
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { videoRef, video, landmarker, delegate, startup, error, retry } = useCameraPose(sel.model);
  const [report, setReport] = useState<SetupReport | null>(null);

  const onFrame = useCallback(
    (r: FrameResult) => {
      const lm = r.result.landmarks[0] ?? null;
      setReport(setupReport(lm, r.width, r.height, view, sel.side));
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) {
        if (canvas.width !== r.width) canvas.width = r.width;
        if (canvas.height !== r.height) canvas.height = r.height;
        drawOverlay(ctx, lm, r.width, r.height, {
          side: sel.side,
          primary: "",
          angle: null,
          target: 0,
          unit: "",
          mirror: true,
          colors: { ...OVERLAY_COLORS, band: "transparent" },
        });
      }
    },
    [view, sel.side],
  );
  usePoseLoop(video, landmarker, onFrame);

  const status =
    startup === "camera" ? "Starting camera…" : startup === "model" ? "Loading pose model…" : startup === "ready" ? `Pose model ready (${delegate})` : "Camera not available";
  const problem = startup === "error" ? describeCameraError(error) : null;
  const checks = setupChecklist(report, view, sel.side);

  return (
    <section>
      <Stepper current={1} />
      <div className="page-head">
        <h1>Setup check</h1>
        <p className="lede">
          {ex ? `${ex.name}, ${sel.side === "R" ? "right" : "left"} arm. ` : ""}
          Get into position so the camera can see you clearly.
        </p>
      </div>
      <div className="setup-layout">
        <div className="stack">
          <div className="stage">
            <video ref={videoRef} className="mirror" />
            <canvas ref={canvasRef} />
            {problem ? (
              <div className="stage-overlay" role="alert">
                <CircleAlert size={36} aria-hidden="true" />
                <h2>{problem.title}</h2>
                <p>{problem.detail}</p>
                <button className="btn" onClick={retry}>Try again</button>
              </div>
            ) : startup === "camera" ? (
              <div className="stage-overlay">
                <LoaderCircle size={36} className="spin" aria-hidden="true" />
                <p>{status}</p>
              </div>
            ) : null}
          </div>
          <p className="setup-status" role="status">
            {startup === "ready" ? <CircleCheck size={18} aria-hidden="true" /> : startup !== "error" ? <LoaderCircle size={18} className="spin" aria-hidden="true" /> : null}
            {status}
          </p>
        </div>
        <aside className="card stack" aria-label="Camera checklist">
          <h2>Checklist</h2>
          <ul className="checklist">
            {checks.map((c) => {
              const Icon = CHECK_ICON[c.state];
              return (
                <li key={c.id} className={`check check-${c.state}`} data-testid={c.id === "joints" ? "joints" : undefined} data-state={c.state}>
                  <Icon size={20} aria-hidden="true" />
                  <span>
                    <span className="sr-only">{CHECK_WORD[c.state]} </span>
                    {c.text}
                  </span>
                </li>
              );
            })}
          </ul>
          <p className="sr-only" role="status">{report?.ready ? "All checks passed. Ready to start." : ""}</p>
          <p className="muted small">
            The skeleton shows what the camera can track; the highlighted lines are the arm being measured. If it
            flickers, stand 2–3 m away in good light with fitted clothing, or choose the heavy pose model.
          </p>
          <p className="note note-critical">
            <HeartPulse size={20} aria-hidden="true" />
            <span>Stop exercising if you feel pain.</span>
          </p>
          <button className="btn btn-primary btn-lg btn-block" disabled={!report?.ready} onClick={() => navigate("/live")}>
            Start session
            <ArrowRight size={20} aria-hidden="true" />
          </button>
          <button className="btn btn-quiet btn-block" onClick={() => navigate("/live")}>Start anyway</button>
        </aside>
      </div>
    </section>
  );
}
