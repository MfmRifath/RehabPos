"""API models: WebSocket protocol v1 (mirrored in frontend/src/ws/protocol.ts) and REST bodies.

Additive optional fields beyond the spec's v1 message shapes (clients may ignore them):
`feedback.cue.severity` (banner colour) and `feedback.dropped` (frames skipped by backpressure).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PROTOCOL_VERSION = 1
Phase = Literal["rest", "ascending", "peak", "descending"]
Severity = Literal["info", "warn", "error"]
SideT = Literal["L", "R"]
ViewT = Literal["sagittal", "frontal"]
UserId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

Landmark4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Landmarks33 = Annotated[list[Landmark4], Field(min_length=33, max_length=33)]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1


# --- client -> server ---------------------------------------------------------------------------


class ImgSize(BaseModel):
    """Image size in pixels."""

    w: int = Field(gt=0)
    h: int = Field(gt=0)


class LandmarksMsg(_Msg):
    """One frame of browser pose output (normalized image + world landmarks)."""

    type: Literal["landmarks"]
    sid: str
    frame: int = Field(ge=0)
    ts_ms: int = Field(ge=0)
    img: ImgSize
    lm: Landmarks33
    world: Landmarks33
    client_infer_ms: float | None = None


class ControlMsg(_Msg):
    """Set control."""

    type: Literal["control"]
    action: Literal["start_set", "pause", "end_set", "calibrate"]


ClientMsg = Annotated[LandmarksMsg | ControlMsg, Field(discriminator="type")]
CLIENT_MSG: TypeAdapter[LandmarksMsg | ControlMsg] = TypeAdapter(ClientMsg)


# --- server -> client ---------------------------------------------------------------------------


class RepInfo(BaseModel):
    """Running rep summary."""

    count: int
    last_peak: float | None = None
    last_duration_s: float | None = None
    last_similarity: float | None = None


class FlagOut(BaseModel):
    """A raised flag."""

    code: str
    severity: Severity
    confidence: float
    source: Literal["rule", "model"]


class CueOut(BaseModel):
    """The single cue to display (speak = say it now)."""

    code: str
    text: str
    speak: bool
    severity: Severity | None = None


class Quality(BaseModel):
    """Setup quality."""

    visibility_ok: bool
    view_ok: bool


class FeedbackMsg(_Msg):
    """Engine output for one processed landmarks message."""

    type: Literal["feedback"] = "feedback"
    frame: int
    ts_ms: int
    angles: dict[str, float | None]
    phase: Phase
    rep: RepInfo
    flags: list[FlagOut]
    cue: CueOut | None
    quality: Quality
    server_ms: float
    dropped: int = 0


class RepOut(BaseModel):
    """A completed repetition."""

    model_config = ConfigDict(from_attributes=True)

    idx: int
    t_start: float
    t_peak: float
    t_end: float
    peak_angle: float
    duration_s: float
    peak_vel: float
    similarity: float
    label: str
    flags: list[str]


class RepCompletedMsg(_Msg):
    """Sent when a rep passes validation."""

    type: Literal["rep_completed"] = "rep_completed"
    rep: RepOut


class SetupStatusMsg(_Msg):
    """Calibration / setup state changes."""

    type: Literal["setup_status"] = "setup_status"
    calibrated: bool
    visibility_ok: bool
    view_ok: bool
    hints: list[str]


class ErrorMsg(_Msg):
    """Protocol or session error (the connection stays open unless the session is invalid)."""

    type: Literal["error"] = "error"
    code: str
    message: str


# --- REST ---------------------------------------------------------------------------------------


class HealthOut(BaseModel):
    """Liveness information."""

    status: Literal["ok"]
    version: str
    pose_model: str
    torch_device: str
    exercises: int


class RuleOut(BaseModel):
    """One exercise rule as exposed to clients."""

    code: str
    kind: Literal["frame", "rep"]
    measure: str | None = None
    op: str | None = None
    value: float | None = None
    severity: Severity
    experimental: bool = False


class ExerciseOut(BaseModel):
    """Exercise definition for clients (defaults; not clinical prescriptions)."""

    id: str
    name: str
    view: ViewT
    allowed_views: list[ViewT]
    experimental: bool
    primary_angle: str
    unit: str
    target: float
    min_rep_s: float
    fsm: dict[str, float]
    rules: list[RuleOut]
    cue_texts: dict[str, str]


class TargetIn(BaseModel):
    """Therapist override for one exercise (None = keep the default)."""

    model_config = ConfigDict(from_attributes=True)

    exercise_id: str
    target_rom: float | None = Field(default=None, gt=0)
    min_rep_s: float | None = Field(default=None, gt=0)
    thresholds: dict[str, float] = Field(default_factory=dict)


class TargetsOut(BaseModel):
    """All overrides for a user."""

    user_id: str
    targets: list[TargetIn]


class SessionCreate(BaseModel):
    """Create a session (the user is created on first use)."""

    model_config = ConfigDict(extra="forbid")

    user_id: UserId
    exercise_id: str
    side: SideT
    view: ViewT | None = None
    source: Literal["live", "upload"] = "live"
    display_name: str | None = Field(default=None, max_length=64)


class SessionOut(BaseModel):
    """Session metadata and summary."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    exercise_id: str
    side: str
    view: str
    source: str
    started_at: datetime
    ended_at: datetime | None
    rep_count: int
    mean_rom: float | None
    quality_score: float | None
    model_version: str


class EventOut(BaseModel):
    """A flag event."""

    model_config = ConfigDict(from_attributes=True)

    t: float
    code: str
    severity: str
    confidence: float
    source: str


class SessionDetail(SessionOut):
    """Session with reps and events."""

    reps: list[RepOut]
    events: list[EventOut]


class JobOut(BaseModel):
    """Background upload-processing job."""

    id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float
    session_id: str
    error: str | None = None
    result: dict[str, str] = Field(default_factory=dict)


class VideoAccepted(BaseModel):
    """Response to a video upload."""

    job_id: str
    session_id: str
