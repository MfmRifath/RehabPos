"""RehabPose: real-time, camera-only rehabilitation exercise coach (research prototype)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rehabpose")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

DISCLAIMER = (
    "RehabPose is a research and portfolio prototype. It is not a medical device, has not been "
    "clinically validated, and must not be used to diagnose, treat, or replace advice from a "
    "qualified healthcare professional. Stop exercising if you feel pain. Thresholds are "
    "engineering defaults, not clinical prescriptions."
)

__all__ = ["DISCLAIMER", "__version__"]
