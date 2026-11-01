import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";
import { setupReport, type SetupReport } from "../lib/setup";
import { startCamera, stopCamera } from "../pose/camera";
import { drawOverlay } from "../pose/draw";
import { getLandmarker } from "../pose/landmarker";
import { usePoseLoop, type FrameResult } from "../pose/usePoseLoop";
import { navigate } from "../router";
import { useApp } from "../store/app";
import { useExercises } from "../useExercises";

export function Setup() {
  const sel = useApp((s) => s.selection);
  const { exercises } = useExercises();
  const ex = exercises.find((e) => e.id === sel.exerciseId);
  const view = sel.view ?? ex?.view ?? "sagittal";
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [landmarker, setLandmarker] = useState<PoseLandmarker | null>(null);
  const [status, setStatus] = useState("Starting camera…");
  const [report, setReport] = useState<SetupReport | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let lm: PoseLandmarker | null;
    let cancelled = false;
    (async () => {
      try {
        const v = videoRef.current as HTMLVideoElement;
        stream = await startCamera(v);
        setVideo(v);
        setStatus("Loading pose model…");
        const created = await getLandmarker(sel.model);
        if (cancelled) return;
        lm = created.landmarker;
        setLandmarker(lm);
        setStatus(`Pose model ready (${created.delegate})`);
      } catch (err) {
        setStatus(`Camera or model error: ${String(err)}`);
      }
    })();
    return () => {
      cancelled = true;
      stopCamera(stream);
    };
  }, [sel.model]);

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
          colors: { bone: "rgba(255,255,255,0.85)", active: "#fab219", band: "transparent", text: "#fff" },
        });
      }
    },
    [view, sel.side],
  );
  usePoseLoop(video, landmarker, onFrame);

  const arm = sel.side === "R" ? "right" : "left";
  const viewText = view === "sagittal" ? `Stand side-on with your ${arm} arm toward the camera` : "Face the camera";
  return (
    <section>
      <h1>Setup check</h1>
      <p className="muted">{status}</p>
      <div className="stage">
        <video ref={videoRef} className="mirror" />
        <canvas ref={canvasRef} />
      </div>
      <ul className="check" aria-live="polite">
        <li className={report?.detected ? "ok" : ""}>Person detected</li>
        <li className={report?.personVisible ? "ok" : ""} data-testid="joints">
          {report?.missing.length ? `Not clearly visible: ${report.missing.join(", ")}` : `Your ${arm} arm and the joints it needs are visible`}
        </li>
        <li className={report?.viewOk ? "ok" : ""}>{viewText}{report?.view ? ` (detected: ${report.view})` : ""}</li>
        <li className={report?.distance === "ok" ? "ok" : ""}>
          Distance {report?.distance === "too_close" ? "– step back" : report?.distance === "too_far" ? "– come closer" : "ok"}
        </li>
      </ul>
      <p className="muted">
        The skeleton shows what the camera can track; the highlighted lines are the arm being measured.
        If it flickers, stand 2–3 m away in good light with fitted clothing, or choose the heavy pose model.
      </p>
      <div className="row">
        <button className="primary" disabled={!report?.ready} onClick={() => navigate("/live")}>
          Start session
        </button>
        <button onClick={() => navigate("/live")}>Start anyway</button>
      </div>
    </section>
  );
}
