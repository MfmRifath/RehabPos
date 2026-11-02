// Camera + pose model startup shared by the Setup and Live screens.
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ModelVariant } from "../store/app";
import { startCamera, stopCamera } from "./camera";
import { getLandmarker, type Delegate } from "./landmarker";

export type PoseStartup = "camera" | "model" | "ready" | "error";

export function useCameraPose(model: ModelVariant) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [landmarker, setLandmarker] = useState<PoseLandmarker | null>(null);
  const [delegate, setDelegate] = useState<Delegate | null>(null);
  const [startup, setStartup] = useState<PoseStartup>("camera");
  const [error, setError] = useState<unknown>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    (async () => {
      try {
        const v = videoRef.current as HTMLVideoElement;
        const started = await startCamera(v);
        if (cancelled) {
          stopCamera(started);
          return;
        }
        stream = started;
        setVideo(v);
        setStartup("model");
        const created = await getLandmarker(model);
        if (cancelled) return;
        setLandmarker(created.landmarker);
        setDelegate(created.delegate);
        setStartup("ready");
      } catch (err) {
        if (cancelled) return;
        setError(err);
        setStartup("error");
      }
    })();
    return () => {
      cancelled = true;
      stopCamera(stream);
    };
  }, [model, attempt]);

  const retry = useCallback(() => {
    setError(null);
    setStartup("camera");
    setAttempt((a) => a + 1);
  }, []);

  return { videoRef, video, landmarker, delegate, startup, error, retry };
}
