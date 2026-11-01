/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

/**
 * Production Content-Security-Policy. `connect-src 'self'` keeps every network request on the
 * RehabPose origin: @mediapipe/tasks-vision otherwise POSTs usage logs to
 * https://odml.pa.googleapis.com/v1/log (found by the e2e smoke test), which would break the
 * "no third-party telemetry" privacy promise (docs/09 §4). Not applied in dev (Vite needs inline
 * scripts for HMR).
 */
export const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "connect-src 'self'",
  "img-src 'self' data: blob:",
  "media-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

const cspPlugin = (): Plugin => ({
  name: "rehabpose-csp",
  apply: "build",
  transformIndexHtml: () => [
    { tag: "meta", attrs: { "http-equiv": "Content-Security-Policy", content: CSP }, injectTo: "head-prepend" },
  ],
});

const API = process.env.REHABPOSE_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), cspPlugin()],
  server: {
    port: 5173,
    proxy: {
      "/api": API,
      "/ws": { target: API.replace(/^http/, "ws"), ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: true, chunkSizeWarningLimit: 1500 },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
