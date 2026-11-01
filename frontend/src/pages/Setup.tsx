import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";
import { setupReport, type SetupReport } from "../lib/setup";
import { startCamera, stopCamera } from "../pose/camera";
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
    (r: FrameResult) => setReport(setupReport(r.result.landmarks[0] ?? null, r.width, r.height, view)),
    [view],
  );
  usePoseLoop(video, landmarker, onFrame);

  const viewText = view === "sagittal" ? "Stand side-on to the camera" : "Face the camera";
  return (
    <section>
      <h1>Setup check</h1>
      <p className="muted">{status}</p>
      <div className="stage">
        <video ref={videoRef} className="mirror" />
      </div>
      <ul className="check" aria-live="polite">
        <li className={report?.personVisible ? "ok" : ""}>Head, shoulders, elbows, wrists and hips visible</li>
        <li className={report?.viewOk ? "ok" : ""}>{viewText}{report?.view ? ` (detected: ${report.view})` : ""}</li>
        <li className={report?.distance === "ok" ? "ok" : ""}>
          Distance {report?.distance === "too_close" ? "– step back" : report?.distance === "too_far" ? "– come closer" : "ok"}
        </li>
      </ul>
      <div className="row">
        <button className="primary" disabled={!report?.ready} onClick={() => navigate("/live")}>
          Start session
        </button>
        <button onClick={() => navigate("/live")}>Start anyway</button>
      </div>
    </section>
  );
}
