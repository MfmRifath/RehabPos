// MediaPipe Tasks Vision PoseLandmarker (VIDEO mode), GPU delegate with CPU fallback.
// Wasm and models are served from public/ (copied by scripts/copy-assets.mjs): no CDN.
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { ModelVariant } from "../store/app";

export type Delegate = "GPU" | "CPU";

const base = import.meta.env.BASE_URL;

export async function createLandmarker(
  variant: ModelVariant,
  preferred: Delegate[] = ["GPU", "CPU"],
): Promise<{ landmarker: PoseLandmarker; delegate: Delegate }> {
  const fileset = await FilesetResolver.forVisionTasks(`${base}mediapipe/wasm`);
  let lastError: unknown = null;
  for (const delegate of preferred) {
    try {
      const landmarker = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${base}models/pose_landmarker_${variant}.task`, delegate },
        runningMode: "VIDEO",
        numPoses: 1,
      });
      return { landmarker, delegate };
    } catch (err) {
      lastError = err;
      console.warn(`PoseLandmarker ${delegate} delegate failed`, err);
    }
  }
  throw new Error(`could not create PoseLandmarker: ${String(lastError)}`);
}

const cache = new Map<ModelVariant, Promise<{ landmarker: PoseLandmarker; delegate: Delegate }>>();

/**
 * Shared landmarker per model variant (setup -> live reuses it instead of reloading the model).
 * Pages must not close it; a failed creation is evicted so it can be retried.
 */
export function getLandmarker(variant: ModelVariant): Promise<{ landmarker: PoseLandmarker; delegate: Delegate }> {
  let p = cache.get(variant);
  if (!p) {
    p = createLandmarker(variant);
    p.catch(() => cache.delete(variant));
    cache.set(variant, p);
  }
  return p;
}
