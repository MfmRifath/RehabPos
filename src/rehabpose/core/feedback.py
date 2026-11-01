"""Cue selection: priority, debounce (>= 0.5 s) and spoken-cue rate limit (1 per 3 s)."""

from __future__ import annotations

from dataclasses import dataclass

from rehabpose.core.exercise import Severity
from rehabpose.core.rules import Flag

SETUP_TEXTS: dict[str, str] = {
    "SETUP_VISIBILITY": "Step back so your whole upper body is visible",
    "SETUP_VIEW_SAGITTAL": "Turn side-on to the camera",
    "SETUP_VIEW_FRONTAL": "Face the camera",
    "CALIBRATING": "Hold still for a moment",
}


@dataclass(frozen=True, slots=True)
class Cue:
    """The single cue to display; speak=True when it should be spoken now."""

    code: str
    text: str
    severity: Severity
    speak: bool


def priority(code: str) -> int:
    """Lower is more urgent: setup > TRUNK_* > SHOULDER_HIKE/PLANE_DRIFT > ELBOW_*/SHOULDER_DRIFT
    > TOO_FAST > INCOMPLETE_ROM > anything else."""
    if code.startswith("SETUP_") or code == "CALIBRATING":
        return 0
    if code.startswith("TRUNK_"):
        return 1
    if code in ("SHOULDER_HIKE", "PLANE_DRIFT"):
        return 2
    if code.startswith("ELBOW_") or code == "SHOULDER_DRIFT":
        return 3
    if code == "TOO_FAST":
        return 4
    if code == "INCOMPLETE_ROM":
        return 5
    return 6


class CueSelector:
    """Chooses at most one cue per frame from the active flags."""

    def __init__(self, debounce_s: float = 0.5, speak_interval_s: float = 3.0) -> None:
        self.debounce_s = debounce_s
        self.speak_interval_s = speak_interval_s
        self.reset()

    def reset(self) -> None:
        """Forget candidate, displayed cue and speech history."""
        self._cand: str | None = None
        self._cand_since = 0.0
        self._shown: str | None = None
        self._pending_speak = False
        self._last_spoken = -float("inf")

    def update(self, t_s: float, flags: list[Flag], texts: dict[str, str]) -> Cue | None:
        """Return the cue to show at time t_s (or None)."""
        if not flags:
            self._cand = self._shown = None
            self._pending_speak = False
            return None
        best = min(flags, key=lambda f: (priority(f.code), -f.confidence, f.code))
        if best.code != self._cand:
            self._cand, self._cand_since = best.code, t_s
        if t_s - self._cand_since < self.debounce_s:
            # keep showing the previous cue only if it is still raised
            if self._shown is not None and any(f.code == self._shown for f in flags):
                shown = next(f for f in flags if f.code == self._shown)
                return self._emit(t_s, shown, texts)
            self._shown = None
            return None
        if best.code != self._shown:
            self._shown = best.code
            self._pending_speak = True
        return self._emit(t_s, best, texts)

    def _emit(self, t_s: float, flag: Flag, texts: dict[str, str]) -> Cue:
        speak = self._pending_speak and t_s - self._last_spoken >= self.speak_interval_s
        if speak:
            self._last_spoken = t_s
            self._pending_speak = False
        text = texts.get(flag.code) or SETUP_TEXTS.get(flag.code, flag.code)
        return Cue(flag.code, text, flag.severity, speak)
