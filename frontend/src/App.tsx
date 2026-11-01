import { lazy, Suspense } from "react";
import { DisclaimerModal } from "./components/DisclaimerModal";
import { Home } from "./pages/Home";
import { Live } from "./pages/Live";
import { Setup } from "./pages/Setup";
import { Summary } from "./pages/Summary";
import { Therapist } from "./pages/Therapist";
import { Upload } from "./pages/Upload";
import { useRoute } from "./router";
import { useApp } from "./store/app";

// recharts is large: load the progress dashboard on demand.
const ProgressPage = lazy(() => import("./pages/Progress").then((m) => ({ default: m.ProgressPage })));

export function App() {
  const route = useRoute();
  const accepted = useApp((s) => s.profile.disclaimerAccepted);
  const accept = useApp((s) => s.acceptDisclaimer);
  const userId = useApp((s) => s.profile.userId);
  const [page, arg] = route;

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/">RehabPose</a>
        <nav aria-label="Main">
          <a href="#/">Exercises</a>
          <a href="#/progress">Progress</a>
          <a href="#/upload">Upload</a>
          <a href="#/therapist">Therapist</a>
        </nav>
        <span className="muted" title="Pseudonymous user id">user {userId}</span>
      </header>
      {!accepted && <DisclaimerModal onAccept={accept} />}
      <main>
        {page === "setup" ? (
          <Setup />
        ) : page === "live" ? (
          <Live />
        ) : page === "summary" && arg ? (
          <Summary sid={arg} />
        ) : page === "progress" ? (
          <Suspense fallback={<p className="muted">Loading…</p>}>
            <ProgressPage />
          </Suspense>
        ) : page === "therapist" ? (
          <Therapist />
        ) : page === "upload" ? (
          <Upload />
        ) : (
          <Home />
        )}
      </main>
    </div>
  );
}
