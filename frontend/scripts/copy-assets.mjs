// Copy MediaPipe wasm runtime and pose models into public/ so the app has no runtime CDN
// dependency (NFR-09). Models come from ../data/models (downloaded with owner consent);
// missing models are reported, not downloaded.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasmSrc = join(root, "node_modules/@mediapipe/tasks-vision/wasm");
const wasmDst = join(root, "public/mediapipe/wasm");
mkdirSync(wasmDst, { recursive: true });
for (const f of readdirSync(wasmSrc)) copyFileSync(join(wasmSrc, f), join(wasmDst, f));
console.log(`copied ${readdirSync(wasmSrc).length} wasm files -> public/mediapipe/wasm`);

const modelsSrc = join(root, "../data/models");
const modelsDst = join(root, "public/models");
mkdirSync(modelsDst, { recursive: true });
for (const v of ["lite", "full", "heavy"]) {
  const name = `pose_landmarker_${v}.task`;
  if (existsSync(join(modelsSrc, name))) {
    copyFileSync(join(modelsSrc, name), join(modelsDst, name));
    console.log(`copied ${name} -> public/models`);
  } else {
    console.warn(`missing ../data/models/${name} (see rehabpose.pose.model_files for the URL)`);
  }
}
