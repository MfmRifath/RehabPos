// Turns thrown errors into plain-language messages people can act on.
import { ApiError } from "./api";

export interface FriendlyError {
  title: string;
  detail: string;
}

/** getUserMedia and model-loading failures. */
export function describeCameraError(err: unknown): FriendlyError {
  const name = err instanceof Error ? err.name : "";
  const message = err instanceof Error ? err.message : String(err);
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      title: "Camera access is blocked",
      detail: "Allow the camera for this page in your browser's site settings, then try again. Video stays in your browser.",
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return { title: "No camera found", detail: "Connect a webcam, or check that your computer can see it, then try again." };
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return { title: "The camera is busy", detail: "Another app may be using it. Close that app, then try again." };
  }
  if (message.includes("Camera API unavailable")) {
    return { title: "The camera needs a secure page", detail: "Open RehabPose over HTTPS, or on localhost on this computer." };
  }
  if (message.includes("PoseLandmarker")) {
    return { title: "The pose model did not load", detail: "Reload the page and try again. If it keeps happening, try the lite model under Advanced on the Exercises screen." };
  }
  return { title: "The camera did not start", detail: message };
}

/** REST failures: the server's own reason when it gave one, otherwise how to get unstuck. */
export function describeApiError(err: unknown): string {
  if (err instanceof ApiError) return err.message || `The server replied with an error (${err.status}).`;
  if (err instanceof TypeError) return "Cannot reach the RehabPose server. Check that it is running, then try again.";
  return err instanceof Error ? err.message : String(err);
}
