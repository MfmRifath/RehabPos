"""WebSocket /ws/session/{sid}: one SessionEngine per connection, latest-frame backpressure."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlmodel import Session

from rehabpose.api.db import Event, Rep, SessionRow, rep_row
from rehabpose.api.schemas import (
    CLIENT_MSG,
    ControlMsg,
    CueOut,
    ErrorMsg,
    FeedbackMsg,
    FlagOut,
    LandmarksMsg,
    Quality,
    RepCompletedMsg,
    RepInfo,
    RepOut,
    SetupStatusMsg,
)
from rehabpose.api.state import AppState
from rehabpose.core.engine import FeedbackFrame, SessionEngine
from rehabpose.core.exercise import ExerciseConfig
from rehabpose.core.feedback import SETUP_TEXTS
from rehabpose.core.landmarks import Frame

log = logging.getLogger(__name__)
router = APIRouter()


class LatestSlot[T]:
    """Backpressure buffer: keeps only the newest unprocessed item; overwrites count as drops."""

    def __init__(self) -> None:
        self._item: T | None = None
        self.dropped = 0

    def put(self, item: T) -> None:
        """Store an item, dropping (and counting) any unprocessed one."""
        if self._item is not None:
            self.dropped += 1
        self._item = item

    def take(self) -> T | None:
        """Remove and return the newest item, if any."""
        item, self._item = self._item, None
        return item


class LiveSession:
    """Synchronous per-connection logic: engine, message building and persistence."""

    def __init__(self, row: SessionRow, exercise: ExerciseConfig, db: Engine) -> None:
        self.sid = row.id
        self.db = db
        self.engine = SessionEngine(exercise, row.side, row.view)  # type: ignore[arg-type]
        self.dropped = 0
        self._t0: float | None = None
        self._last_ts = -1
        self._events_saved = 0
        self._setup: tuple[bool, bool, bool] | None = None

    def handle_landmarks(self, msg: LandmarksMsg, t_recv: float) -> list[str]:
        """Process one frame; returns JSON messages to send (feedback first)."""
        if msg.ts_ms <= self._last_ts:
            err = ErrorMsg(code="non_monotonic_ts", message=f"ts_ms {msg.ts_ms} <= {self._last_ts}")
            return [err.model_dump_json()]
        self._last_ts = msg.ts_ms
        if self._t0 is None:
            self._t0 = msg.ts_ms / 1000.0
        lm, world = np.asarray(msg.lm), np.asarray(msg.world)
        frame = Frame(lm, world, msg.ts_ms, msg.img.w, msg.img.h, msg.frame)
        fb = self.engine.process(frame)
        out = [self._feedback(fb, t_recv).model_dump_json()]
        if fb.completed_rep is not None:
            row = rep_row(self.sid, fb.completed_rep, self._t0)
            out.append(RepCompletedMsg(rep=RepOut.model_validate(row)).model_dump_json())
            self._persist(row)
        else:
            self._persist(None)
        setup = (fb.calibrated, fb.visibility_ok, fb.view_ok)
        if setup != self._setup:
            self._setup = setup
            hints = [SETUP_TEXTS[f.code] for f in fb.flags if f.code in SETUP_TEXTS]
            out.append(
                SetupStatusMsg(
                    calibrated=fb.calibrated,
                    visibility_ok=fb.visibility_ok,
                    view_ok=fb.view_ok,
                    hints=hints,
                ).model_dump_json()
            )
        return out

    def handle_control(self, msg: ControlMsg) -> None:
        """Apply a set-control action."""
        actions = {
            "start_set": self.engine.start_set,
            "pause": self.engine.pause,
            "end_set": self.engine.end_set,
            "calibrate": self.engine.calibrate,
        }
        actions[msg.action]()

    def _feedback(self, fb: FeedbackFrame, t_recv: float) -> FeedbackMsg:
        last = fb.last_rep
        return FeedbackMsg(
            frame=fb.frame_idx,
            ts_ms=fb.ts_ms,
            angles={k: (round(v, 2) if v is not None else None) for k, v in fb.angles.items()},
            phase=fb.phase,
            rep=RepInfo(
                count=fb.rep_count,
                last_peak=last.peak_angle if last else None,
                last_duration_s=last.duration_s if last else None,
                last_similarity=last.dtw_similarity if last else None,
            ),
            flags=[
                FlagOut(
                    code=f.code,
                    severity=f.severity,
                    confidence=round(f.confidence, 3),
                    source=f.source,
                )
                for f in fb.flags
            ],
            cue=(
                CueOut(
                    code=fb.cue.code, text=fb.cue.text, speak=fb.cue.speak, severity=fb.cue.severity
                )
                if fb.cue
                else None
            ),
            quality=Quality(visibility_ok=fb.visibility_ok, view_ok=fb.view_ok),
            server_ms=round((time.perf_counter() - t_recv) * 1000.0, 3),
            dropped=self.dropped,
        )

    def _persist(self, rep: Rep | None) -> None:
        new_events = self.engine.events[self._events_saved :]
        if rep is None and not new_events:
            return
        t0 = self._t0 or 0.0
        with Session(self.db) as db:
            if rep is not None:
                db.add(rep)
            for e in new_events:
                db.add(
                    Event(
                        session_id=self.sid,
                        t=round(e.t - t0, 3),
                        code=e.code,
                        severity=e.severity,
                        confidence=e.confidence,
                        source=e.source,
                    )
                )
            db.commit()
        self._events_saved = len(self.engine.events)


@router.websocket("/ws/session/{sid}")
async def ws_session(ws: WebSocket, sid: str) -> None:
    """Live landmarks in, feedback out (protocol v1)."""
    state: AppState = ws.app.state.rp
    await ws.accept()
    with Session(state.db) as db:
        row = db.get(SessionRow, sid)
        if row is None or row.ended_at is not None:
            reason = "unknown_session" if row is None else "session_ended"
            await ws.send_text(ErrorMsg(code=reason, message=f"session {sid}").model_dump_json())
            await ws.close(code=4404)
            return
        live = LiveSession(row, state.exercise_for(db, row.user_id, row.exercise_id), state.db)

    slot: LatestSlot[tuple[LandmarksMsg, float]] = LatestSlot()
    wake = asyncio.Event()
    closed = False

    async def send_all(msgs: list[str]) -> None:
        for m in msgs:
            await ws.send_text(m)

    async def process_pending() -> None:
        item = slot.take()
        if item is not None:
            live.dropped = slot.dropped
            await send_all(live.handle_landmarks(*item))

    async def receiver() -> None:
        nonlocal closed
        try:
            while True:
                text = await ws.receive_text()
                t_recv = time.perf_counter()
                try:
                    msg = CLIENT_MSG.validate_json(text)
                except ValidationError as exc:
                    first = exc.errors()[0]
                    detail = f"{'.'.join(map(str, first['loc']))}: {first['msg']}"
                    await ws.send_text(
                        ErrorMsg(code="bad_message", message=detail).model_dump_json()
                    )
                    continue
                if isinstance(msg, ControlMsg):
                    await process_pending()  # keep ordering: the frame before the control
                    live.handle_control(msg)
                    continue
                if msg.sid != sid:
                    await ws.send_text(
                        ErrorMsg(code="sid_mismatch", message=msg.sid).model_dump_json()
                    )
                    continue
                slot.put((msg, t_recv))
                wake.set()
        except WebSocketDisconnect:
            pass
        finally:
            closed = True
            wake.set()

    async def processor() -> None:
        while True:
            await wake.wait()
            wake.clear()
            await process_pending()
            if closed:
                return

    proc = asyncio.create_task(processor())
    try:
        await receiver()
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await proc
    finally:
        proc.cancel()
        log.info(
            "ws session %s closed: %d reps, %d dropped", sid, len(live.engine.reps), live.dropped
        )
