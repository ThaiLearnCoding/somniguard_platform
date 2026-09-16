from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class StepOut(BaseModel):
    timestamp_sec: float
    spo2: float
    bpm: float
    motion_level: float
    model_score: Optional[float] = None
    model_label: Optional[int] = None
    human_label: Optional[int] = None
    effective_label: Optional[int] = None
    label_source: str


class SessionSummary(BaseModel):
    id: int
    filename: str
    recording_date: str  # ISO "YYYY-MM-DD" -- the night this recording is for
    uploaded_at: str
    status: str
    n_steps: int
    n_segments: int
    label_counts: Dict[str, int] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    session: SessionSummary
    warnings: List[str] = Field(default_factory=list)


class TimelineResponse(BaseModel):
    session: SessionSummary
    steps: List[StepOut]
    warnings: List[str] = Field(default_factory=list)


class RelabelRequest(BaseModel):
    start_time: float
    end_time: float
    new_label: int = Field(..., ge=0, le=2)
    note: Optional[str] = None


class RelabelResponse(BaseModel):
    n_steps_affected: int
    session: SessionSummary


class TrainRequest(BaseModel):
    session_ids: Optional[List[int]] = None  # None = use all sessions with any human/model labels
    preview_session_id: Optional[int] = None  # None = most recently uploaded session
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    early_stopping_patience: Optional[int] = None


class ModelVersionOut(BaseModel):
    id: int
    name: str
    created_at: str
    parent_id: Optional[int]
    status: str
    version_dir: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    gbl_path: Optional[str] = None
    gbl_build_info: Dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    model_version: ModelVersionOut
    new_metrics: Dict[str, Any]
    parent_metrics: Dict[str, Any]
    n_train_windows: int
    n_test_windows: int
    candidate_preview: Optional[TimelineResponse] = None


class ApproveResponse(BaseModel):
    model_version: ModelVersionOut
    firmware: Dict[str, Any]


class BuildFirmwareResponse(BaseModel):
    model_version: ModelVersionOut
    firmware: Dict[str, Any]
