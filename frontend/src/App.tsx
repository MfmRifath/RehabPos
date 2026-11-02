import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { AppShell } from "./components/AppShell";
import { DisclaimerModal } from "./components/DisclaimerModal";
import { Home } from "./pages/Home";
import { Live } from "./pages/Live";
import { ProgressLoading } from "./pages/ProgressLoading";
import { Setup } from "./pages/Setup";
import { Summary } from "./pages/Summary";
import { Therapist } from "./pages/Therapist";
import { Upload } from "./pages/Upload";
import { useRoute } from "./router";
import { useApp } from "./store/app";

// recharts is large: load the progress dashboard on demand.
const ProgressPage = lazy(() => import("./pages/Progress").then((m) => ({ default: m.ProgressPage })));

const TITLES: Record<string, string> = {
  setup: "Setup check",
  live: "Live session",
  summary: "Session summary",
  progress: "Progress",
  therapist: "Therapist settings",
  upload: "Analyze a video",
};

export function App() {
  const route = useRoute();
  const accepted = useApp((s) => s.profile.disclaimerAccepted);
  const accept = useApp((s) => s.acceptDisclaimer);
  const [reread, setReread] = useState(false);
  const [routed, arg] = route;
  // FR-20: a deep link must not start the camera or a session before the disclaimer is acknowledged.
  const page = !accepted && (routed === "setup" || routed === "live") ? "" : (routed ?? "");

  const key = route.join("/");
  const firstRender = useRef(true);
  useEffect(() => {
    document.title = TITLES[page] ? `${TITLES[page]} · RehabPose` : "RehabPose";
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    window.scrollTo(0, 0);
    document.querySelector<HTMLElement>("main")?.focus({ preventScroll: true });
  }, [key, page]);

  const disclaimer = (!accepted || reread) && (
    <DisclaimerModal
      onAccept={() => {
        accept();
        setReread(false);
      }}
    />
  );

  // The live session is full-bleed: no navigation competes with the camera.
  if (page === "live") return <Live />;

  return (
    <AppShell page={page} onShowDisclaimer={() => setReread(true)}>
      {disclaimer}
      {page === "setup" ? (
        <Setup />
      ) : page === "summary" && arg ? (
        <Summary sid={arg} />
      ) : page === "progress" ? (
        <Suspense fallback={<ProgressLoading />}>
          <ProgressPage />
        </Suspense>
      ) : page === "therapist" ? (
        <Therapist />
      ) : page === "upload" ? (
        <Upload />
      ) : (
        <Home />
      )}
    </AppShell>
  );
}
