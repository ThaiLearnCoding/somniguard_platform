"""
database.py

SQLite schema for the single-patient PoC.

Design note on "sessions" vs "windows": we store one row per 0.5s FEATURE STEP
(steps table), not per 30s window. Windows overlap heavily (stride 0.5s over a
30s/60-step lookback), so storing per-window would duplicate ~98% of the data.
Windows are reconstructed on demand (see ml_utils.build_windows_from_steps) by
slicing 60 consecutive steps. This also makes human relabeling natural: the
doctor corrects a *time range*, which maps directly onto a contiguous run of
step rows.
"""

import json
import datetime as dt
from typing import List, Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime, Date, ForeignKey, Boolean, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session as OrmSession

Base = declarative_base()


class RecordingSession(Base):
    """One uploaded CSV. May internally contain multiple DSP 'segments' (session
    boundary markers in the raw CSV), tracked on the Step rows."""
    __tablename__ = "recording_sessions"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)  # original filename only -- the raw CSV itself is never
                                                # persisted (processed via a temp file, then discarded);
                                                # only the derived per-step feature vectors below are kept.
    recording_date = Column(Date, nullable=False, index=True)  # the NIGHT this recording is for (from
                                                # the CSV's Date column, dd/mm/yy; falls back to the
                                                # ingestion day if the column is absent). This -- not
                                                # uploaded_at -- is what the doctor's day picker filters on.
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)
    status = Column(String, default="uploaded")  # uploaded -> reviewed -> trained
    n_steps = Column(Integer, default=0)
    n_segments = Column(Integer, default=0)
    note = Column(Text, nullable=True)

    steps = relationship("Step", back_populates="session", cascade="all, delete-orphan")
    relabel_events = relationship("RelabelEvent", back_populates="session", cascade="all, delete-orphan")


class Step(Base):
    """One 0.5s feature step (28-dim vector) belonging to a recording session."""
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("recording_sessions.id"), nullable=False, index=True)
    segment_index = Column(Integer, nullable=False)
    step_index = Column(Integer, nullable=False)  # index within its segment
    timestamp_sec = Column(Float, nullable=False)

    spo2 = Column(Float, nullable=False)
    bpm = Column(Float, nullable=False)
    motion_level = Column(Float, nullable=False)
    ir_norm_json = Column(Text, nullable=False)  # JSON list of 25 floats

    model_score = Column(Float, nullable=True)   # raw severity score in [0,1], from the model active at predict time
    model_label = Column(Integer, nullable=True)  # 0/1/2
    human_label = Column(Integer, nullable=True)  # 0/1/2, overrides model_label when set
    label_source = Column(String, default="model")  # 'model' | 'human'
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    session = relationship("RecordingSession", back_populates="steps")

    def feature_vector(self) -> List[float]:
        return [self.spo2, self.bpm] + json.loads(self.ir_norm_json) + [self.motion_level]

    def effective_label(self) -> Optional[int]:
        return self.human_label if self.human_label is not None else self.model_label


class RelabelEvent(Base):
    """Audit trail: every doctor correction, as a (time range -> new label) event."""
    __tablename__ = "relabel_events"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("recording_sessions.id"), nullable=False, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    new_label = Column(Integer, nullable=False)
    n_steps_affected = Column(Integer, default=0)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    session = relationship("RecordingSession", back_populates="relabel_events")


class ModelVersion(Base):
    """One trained model artifact set (float .h5 + int8 .tflite + config.json) on disk."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    parent_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    status = Column(String, default="pending_review")  # pending_review | active | archived | rejected
    version_dir = Column(String, nullable=False)
    metrics_json = Column(Text, nullable=True)          # {"new_metrics": {...}, "parent_metrics": {...}, ...}
    training_params_json = Column(Text, nullable=True)
    gbl_path = Column(String, nullable=True)
    gbl_build_info_json = Column(Text, nullable=True)   # {"mode": "real"|"mock", "success": bool, "log": "..."}

    def metrics(self) -> dict:
        return json.loads(self.metrics_json) if self.metrics_json else {}


_engine = None
_SessionLocal = None


def init_db(db_path: str = "data/somniguard.db"):
    global _engine, _SessionLocal
    _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_db() -> OrmSession:
    if _SessionLocal is None:
        init_db()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> OrmSession:
    """Non-generator helper for use outside FastAPI's Depends() (e.g. in scripts)."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()
