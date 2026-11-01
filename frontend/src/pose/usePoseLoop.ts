import type { PoseLandmarker, PoseLandmarkerResult } from "@mediapipe/tasks-vision";
import { useEffect, useRef } from "react";

export interface FrameResult {
  result: PoseLandmarkerResult;
  tsMs: number;
  inferMs: number;
  fps: number;
  width: number;
  height: number;
}

/**
 * Run detectForVideo once per new camera frame with strictly increasing timestamps.
 * Uses requestVideoFrameCallback (fires per presented frame); for a live MediaStream
 * `currentTime` follows the wall clock, so the requestAnimationFrame fallback can only
 * de-duplicate approximately. `fps` = processed camera frames per second.
 */
export function usePoseLoop(
  video: HTMLVideoElement | null,
  landmarker: PoseLandmarker | null,
  onFrame: (r: FrameResult) => void,
): void {
  const cb = useRef(onFrame);
  useEffect(() => {
    cb.current = onFrame;
  }, [onFrame]);

  useEffect(() => {
    if (!video || !landmarker) return;
    let stopped = false;
    let lastTs = -1;
    let lastVideoTime = -1;
    let lastFrameAt = performance.now();
    let fps = 0;

    const process = () => {
      if (video.readyState < 2 || video.videoWidth === 0) return;
      const now = performance.now();
      const tsMs = Math.max(Math.round(now), lastTs + 1);
      lastTs = tsMs;
      const result = landmarker.detectForVideo(video, tsMs);
      const inferMs = performance.now() - now;
      const inst = 1000 / Math.max(now - lastFrameAt, 1);
      lastFrameAt = now;
      fps = fps === 0 ? inst : 0.9 * fps + 0.1 * inst;
      cb.current({ result, tsMs, inferMs, fps, width: video.videoWidth, height: video.videoHeight });
    };

    // lib.dom declares rVFC as always present; older Firefox lacks it, so check at runtime.
    const hasRvfc = typeof (video as { requestVideoFrameCallback?: unknown }).requestVideoFrameCallback === "function";
    if (hasRvfc) {
      let handle = 0;
      const onVideoFrame = () => {
        if (stopped) return;
        process();
        handle = video.requestVideoFrameCallback(onVideoFrame);
      };
      handle = video.requestVideoFrameCallback(onVideoFrame);
      return () => {
        stopped = true;
        video.cancelVideoFrameCallback(handle);
      };
    }
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (video.currentTime === lastVideoTime) return;
      lastVideoTime = video.currentTime;
      process();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [video, landmarker]);
}
