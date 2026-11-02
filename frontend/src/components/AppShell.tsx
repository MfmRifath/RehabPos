import { Dumbbell, Stethoscope, TrendingUp, Upload } from "lucide-react";
import type { ReactNode } from "react";
import { BrandMark } from "./ui";

const NAV = [
  { href: "#/", label: "Exercises", icon: Dumbbell, pages: ["", "setup", "summary"] },
  { href: "#/progress", label: "Progress", icon: TrendingUp, pages: ["progress"] },
  { href: "#/upload", label: "Upload", icon: Upload, pages: ["upload"] },
  { href: "#/therapist", label: "Therapist", icon: Stethoscope, pages: ["therapist"] },
];

export function AppShell({ page, onShowDisclaimer, children }: { page: string; onShowDisclaimer: () => void; children: ReactNode }) {
  return (
    <div className="app">
      <a className="skip-link" href="#main" onClick={(e) => {
        // A plain #main link would change the hash route.
        e.preventDefault();
        document.querySelector("main")?.focus();
      }}>
        Skip to content
      </a>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#/">
            <BrandMark />
            RehabPose
          </a>
          <nav className="nav" aria-label="Main">
            {NAV.map(({ href, label, icon: Icon, pages }) => (
              <a key={href} href={href} aria-current={pages.includes(page) ? "page" : undefined}>
                <Icon size={18} aria-hidden="true" />
                {label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="content" tabIndex={-1}>
        {children}
      </main>
      <footer className="footer">
        <span>Research prototype. Not a medical device.</span>
        <button className="link-button" onClick={onShowDisclaimer}>Read the full disclaimer</button>
      </footer>
    </div>
  );
}
