"""
main.py - FastAPI backend for the SomniGuard doctor-in-the-loop platform.

Ingestion model: this endpoint (/api/upload) is how a night's recording gets INTO the
platform. In production that will be the mobile app pushing data automatically each
morning once it has Wi-Fi -- there is deliberately no "upload" control in the doctor's
UI, because by the time the doctor looks at the dashboard, the platform already has the
data. For now (before that mobile integration exists), use scripts/ingest_recording.py
to push CSVs into a running instance the same way the mobile app eventually will.

Model lifecycle (models/current is always THE deployed model; no duplication, nothing lost):
    models/current/   <- the ACTIVE model. Used directly for all inference. Single source of truth.
    models/staging/   <- a fine-tuned CANDIDATE awaiting the doctor's approve decision. Single slot,
                          cleared and rewritten by every /api/train call -- never accumulates.
    models/history/<name>/  <- a SNAPSHOT of a model that USED TO be active, taken at the moment it's
                          replaced (i.e. right before models/current/ is overwritten by an approved
                          candidate). This is the only place old versions live, so nothing is ever lost
                          (including the very first, hand-provided model) and nothing is ever duplicated
                          (the currently-active files exist in exactly one place: models/current/).

Endpoints (see README.md for the full walkthrough):
    POST /api/models/register-initial   bootstrap: register models/current/ as the active model
    POST /api/upload                    ingest a raw CSV for one night (not persisted as a file), run
                                         DSP + active model. Not called from the doctor UI -- see above.
    GET  /api/sessions                  list ingested nights (each with its recording_date)
    GET  /api/sessions/{id}/timeline    per-timestep data for the doctor's chart (always live-refreshed
                                         against whichever model is CURRENTLY active)
    POST /api/sessions/{id}/relabel     doctor corrects a time range -- edits that night's data IN
                                         PLACE, never creates a new dataset/session
    POST /api/train                     fine-tune the active model on corrected data; returns a live
                                         preview of the new candidate's own inference for comparison
    GET  /api/models                    list model versions (active/pending/archived) -- API only, not
                                         surfaced in the simplified UI
    POST /api/models/{id}/approve       promote the pending candidate to active (archive-then-replace)
    POST /api/models/{id}/build-firmware  (re)run the .tflite -> .gbl conversion
    GET  /api/download/latest-model     serve the current .gbl (or .tflite) for OTA pull
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must precede any tensorflow import

import json
import shutil
import tempfile
import datetime as dt
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy import func

import database as db
import preprocessing as pp
import ml_utils as mu
import gbl_bridge
import schemas as sch

APP_CONFIG_PATH = os.environ.get("SOMNIGUARD_APP_CONFIG", "config/app_config.json")


def load_app_config() -> dict:
    p = Path(APP_CONFIG_PATH)
    if not p.exists():
        raise RuntimeError(f"App config not found at {p}. Copy/edit config/app_config.json.")
    with open(p) as f:
        return json.load(f)


CFG = load_app_config()
MODELS_CURRENT_DIR = Path(CFG["models_current_dir"])
MODELS_HISTORY_DIR = Path(CFG["models_history_dir"])
MODELS_STAGING_DIR = Path(CFG["models_staging_dir"])
PPG_SAMPLE_RATE = CFG.get("ppg_sample_rate_hz", 50)
TRAIN_DEFAULTS = CFG.get("training_defaults", {})
MODEL_FILES = ("model_float.h5", "model_int8.tflite", "config.json")

for d in (MODELS_CURRENT_DIR, MODELS_HISTORY_DIR, MODELS_STAGING_DIR, Path(CFG["db_path"]).parent):
    d.mkdir(parents=True, exist_ok=True)

db.init_db(CFG["db_path"])

app = FastAPI(title="SomniGuard AutoML & Human-in-the-Loop Platform")


def get_db():
    yield from db.get_db()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _model_version_to_out(row: db.ModelVersion) -> sch.ModelVersionOut:
    return sch.ModelVersionOut(
        id=row.id, name=row.name, created_at=row.created_at.isoformat(),
        parent_id=row.parent_id, status=row.status, version_dir=row.version_dir,
        metrics=row.metrics(),
        gbl_path=row.gbl_path,
        gbl_build_info=json.loads(row.gbl_build_info_json) if row.gbl_build_info_json else {},
    )


def _get_active_model_row(orm: OrmSession) -> Optional[db.ModelVersion]:
    return orm.query(db.ModelVersion).filter(db.ModelVersion.status == "active").one_or_none()


def _session_summary(orm: OrmSession, s: db.RecordingSession) -> sch.SessionSummary:
    counts_q = (
        orm.query(db.Step.human_label, db.Step.model_label, func.count(db.Step.id))
        .filter(db.Step.session_id == s.id)
        .group_by(db.Step.human_label, db.Step.model_label)
        .all()
    )
    label_counts = {"0": 0, "1": 0, "2": 0, "unlabeled": 0}
    for human_label, model_label, cnt in counts_q:
        eff = human_label if human_label is not None else model_label
        if eff is None:
            label_counts["unlabeled"] += cnt
        else:
            label_counts[str(eff)] += cnt
    return sch.SessionSummary(
        id=s.id, filename=s.filename, recording_date=s.recording_date.isoformat(),
        uploaded_at=s.uploaded_at.isoformat(),
        status=s.status, n_steps=s.n_steps, n_segments=s.n_segments,
        label_counts=label_counts,
    )


def _predict_over_session_steps(
    orm: OrmSession, session_id: int, ml_model: mu.ModelVersion
) -> Tuple[List[db.Step], Dict[int, Optional[Tuple[float, int]]]]:
    """Runs `ml_model` (any ModelVersion -- active OR an unreviewed candidate) over one session's
    steps, WITHOUT writing anything to the DB. Returns (ordered steps, {step.id: (score,label) or None}).
    """
    steps = (
        orm.query(db.Step)
        .filter(db.Step.session_id == session_id)
        .order_by(db.Step.segment_index, db.Step.step_index)
        .all()
    )
    if not steps:
        return steps, {}
    window_frames = ml_model.channels["window_frames"]
    features_by_segment: Dict[int, list] = {}
    steps_by_segment: Dict[int, list] = {}
    for s in steps:
        features_by_segment.setdefault(s.segment_index, []).append(s.feature_vector())
        steps_by_segment.setdefault(s.segment_index, []).append(s)
    predictions = mu.predict_over_segments(ml_model, features_by_segment, window_frames)
    result: Dict[int, Optional[Tuple[float, int]]] = {}
    for seg_idx, preds in predictions.items():
        for step_row, pred in zip(steps_by_segment[seg_idx], preds):
            result[step_row.id] = pred
    return steps, result


def _refresh_active_predictions(orm: OrmSession, session_id: int) -> Optional[str]:
    """Runs the CURRENTLY ACTIVE model over a session and writes the result back onto the Step rows
    (so the timeline always reflects whichever model is deployed right now, even if it changed since
    upload). Returns a warning string on failure/no-active-model, else None."""
    active_row = _get_active_model_row(orm)
    if active_row is None:
        return (
            "No active model version is registered yet, so no predictions are shown. "
            "POST /api/models/register-initial once your model files are in models/current/."
        )
    try:
        ml_model = mu.ModelVersion(active_row.version_dir)
        steps, preds = _predict_over_session_steps(orm, session_id, ml_model)
    except mu.MLError as e:
        return f"Active model inference failed: {e}"

    for s in steps:
        pred = preds.get(s.id)
        if pred is not None:
            s.model_score, s.model_label = pred
    orm.commit()
    return None


def _step_to_out(s: db.Step) -> sch.StepOut:
    return sch.StepOut(
        timestamp_sec=s.timestamp_sec, spo2=s.spo2, bpm=s.bpm, motion_level=s.motion_level,
        model_score=s.model_score, model_label=s.model_label, human_label=s.human_label,
        effective_label=s.effective_label(), label_source=s.label_source,
    )


# --------------------------------------------------------------------------------------
# Bootstrap: register the model you drop into models/current/
# --------------------------------------------------------------------------------------

@app.post("/api/models/register-initial", response_model=sch.ModelVersionOut)
def register_initial_model(force: bool = False, orm: OrmSession = Depends(get_db)):
    existing_active = _get_active_model_row(orm)
    if existing_active is not None and not force:
        raise HTTPException(
            409,
            f"An active model version already exists (id={existing_active.id}, "
            f"name='{existing_active.name}'). Pass ?force=true to replace it, or use "
            f"POST /api/train + /api/models/{{id}}/approve for the normal flow.",
        )

    try:
        mu.ModelVersion(str(MODELS_CURRENT_DIR))
    except mu.MLError as e:
        raise HTTPException(400, f"Could not load models/current/: {e}")

    if existing_active is not None:
        existing_active.status = "archived"

    row = db.ModelVersion(
        name=f"initial-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        parent_id=None, status="active", version_dir=str(MODELS_CURRENT_DIR),
        metrics_json=json.dumps({"note": "Initial model registered from models/current/, no fine-tune metrics yet."}),
    )
    orm.add(row)
    orm.commit()
    orm.refresh(row)
    return _model_version_to_out(row)


# --------------------------------------------------------------------------------------
# Phase 1: Upload (raw CSV is processed via a temp file and immediately discarded --
# only the derived per-step feature vectors are ever persisted) + initial prediction
# --------------------------------------------------------------------------------------

@app.post("/api/upload", response_model=sch.UploadResponse)
def upload_csv(file: UploadFile = File(...), orm: OrmSession = Depends(get_db)):
    warnings: List[str] = []

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        try:
            step_features = pp.compute_step_features(tmp.name, ppg_sample_rate=PPG_SAMPLE_RATE)
        except pp.PreprocessingError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(400, f"Failed to parse/process CSV: {e}")

        recording_date = pp.parse_recording_date(tmp.name)
        if recording_date is None:
            recording_date = dt.datetime.utcnow().date()
            warnings.append(
                "No (parseable) Date column found in this CSV; using today's date "
                f"({recording_date.isoformat()}) instead. Add a Date column (dd/mm/yy) "
                "to the export for correct multi-night tracking."
            )
    finally:
        Path(tmp.name).unlink(missing_ok=True)  # never keep the raw upload around

    n_segments = len({sf.segment_index for sf in step_features})
    session_row = db.RecordingSession(
        filename=file.filename, recording_date=recording_date, status="uploaded",
        n_steps=len(step_features), n_segments=n_segments,
    )
    orm.add(session_row)
    orm.commit()
    orm.refresh(session_row)

    step_rows = [
        db.Step(
            session_id=session_row.id, segment_index=sf.segment_index, step_index=sf.step_index,
            timestamp_sec=sf.timestamp_sec, spo2=sf.spo2, bpm=sf.bpm, motion_level=sf.motion_level,
            ir_norm_json=json.dumps(sf.ir_norm), label_source="model",
        )
        for sf in step_features
    ]
    orm.add_all(step_rows)
    orm.commit()

    warn = _refresh_active_predictions(orm, session_row.id)
    if warn:
        warnings.append(warn)

    orm.refresh(session_row)
    return sch.UploadResponse(session=_session_summary(orm, session_row), warnings=warnings)


# --------------------------------------------------------------------------------------
# Phase 2: Dashboard data + relabeling
# --------------------------------------------------------------------------------------

@app.get("/api/sessions", response_model=List[sch.SessionSummary])
def list_sessions(orm: OrmSession = Depends(get_db)):
    rows = (
        orm.query(db.RecordingSession)
        .order_by(db.RecordingSession.recording_date.desc(), db.RecordingSession.uploaded_at.desc())
        .all()
    )
    return [_session_summary(orm, s) for s in rows]


@app.get("/api/sessions/{session_id}/timeline", response_model=sch.TimelineResponse)
def get_timeline(session_id: int, orm: OrmSession = Depends(get_db)):
    session_row = orm.get(db.RecordingSession, session_id)
    if session_row is None:
        raise HTTPException(404, "Session not found")

    warning = _refresh_active_predictions(orm, session_id)  # always reflects whatever's active NOW

    steps = (
        orm.query(db.Step)
        .filter(db.Step.session_id == session_id)
        .order_by(db.Step.segment_index, db.Step.step_index)
        .all()
    )
    step_outs = [_step_to_out(s) for s in steps]
    warnings = [warning] if warning else []
    return sch.TimelineResponse(session=_session_summary(orm, session_row), steps=step_outs, warnings=warnings)


@app.post("/api/sessions/{session_id}/relabel", response_model=sch.RelabelResponse)
def relabel_session(session_id: int, req: sch.RelabelRequest, orm: OrmSession = Depends(get_db)):
    session_row = orm.get(db.RecordingSession, session_id)
    if session_row is None:
        raise HTTPException(404, "Session not found")
    if req.end_time <= req.start_time:
        raise HTTPException(400, "end_time must be after start_time")

    affected = (
        orm.query(db.Step)
        .filter(
            db.Step.session_id == session_id,
            db.Step.timestamp_sec >= req.start_time,
            db.Step.timestamp_sec <= req.end_time,
        )
        .all()
    )
    if not affected:
        raise HTTPException(400, "No steps found in that time range.")

    for s in affected:
        s.human_label = req.new_label
        s.label_source = "human"

    event = db.RelabelEvent(
        session_id=session_id, start_time=req.start_time, end_time=req.end_time,
        new_label=req.new_label, n_steps_affected=len(affected), note=req.note,
    )
    orm.add(event)
    session_row.status = "reviewed"
    orm.commit()
    orm.refresh(session_row)

    return sch.RelabelResponse(n_steps_affected=len(affected), session=_session_summary(orm, session_row))


# --------------------------------------------------------------------------------------
# Phase 3: Fine-tuning (writes to the single staging slot; returns a live preview of the
# candidate's own inference on the session the doctor is looking at, for comparison)
# --------------------------------------------------------------------------------------

@app.post("/api/train", response_model=sch.TrainResponse)
def train_new_model(req: sch.TrainRequest, orm: OrmSession = Depends(get_db)):
    active_row = _get_active_model_row(orm)
    if active_row is None:
        raise HTTPException(
            400,
            "No active model version. POST /api/models/register-initial first "
            "(after placing model_float.h5 / model_int8.tflite / config.json in models/current/).",
        )

    query = orm.query(db.Step)
    if req.session_ids:
        query = query.filter(db.Step.session_id.in_(req.session_ids))
    steps = query.order_by(db.Step.session_id, db.Step.segment_index, db.Step.step_index).all()
    if not steps:
        raise HTTPException(400, "No steps found for the given session_ids.")

    steps_by_segment = {}
    for s in steps:
        key = (s.session_id, s.segment_index)
        steps_by_segment.setdefault(key, []).append((s.feature_vector(), s.effective_label()))

    try:
        parent = mu.ModelVersion(active_row.version_dir)
        window_frames = parent.channels["window_frames"]
        X_w, y_w, _ = mu.build_windows_from_steps(steps_by_segment, window_frames)
    except mu.MLError as e:
        raise HTTPException(400, f"Could not build training windows: {e}")

    params = dict(TRAIN_DEFAULTS)
    for k in ("learning_rate", "epochs", "batch_size", "early_stopping_patience"):
        v = getattr(req, k)
        if v is not None:
            params[k] = v

    # Single reusable staging slot: discard any earlier not-yet-approved candidate (both its DB
    # row and its files) so nothing accumulates across repeated training rounds.
    for p in orm.query(db.ModelVersion).filter(db.ModelVersion.status == "pending_review").all():
        orm.delete(p)
    orm.commit()
    if MODELS_STAGING_DIR.exists():
        shutil.rmtree(MODELS_STAGING_DIR)
    MODELS_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    version_name = f"v-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    try:
        result = mu.finetune(
            parent, X_w, y_w, output_dir=str(MODELS_STAGING_DIR),
            learning_rate=params.get("learning_rate", 1e-5),
            epochs=params.get("epochs", 200),
            batch_size=params.get("batch_size", 32),
            early_stopping_patience=params.get("early_stopping_patience", 15),
        )
    except mu.MLError as e:
        raise HTTPException(400, f"Fine-tuning failed: {e}")

    row = db.ModelVersion(
        name=version_name, parent_id=active_row.id, status="pending_review",
        version_dir=str(MODELS_STAGING_DIR),
        metrics_json=json.dumps({
            "new_metrics": result["new_metrics"], "parent_metrics": result["parent_metrics"],
            "n_train_windows": int(len(X_w)) - result["n_test_windows"], "n_test_windows": result["n_test_windows"],
        }),
        training_params_json=json.dumps(params),
    )
    orm.add(row)
    orm.commit()
    orm.refresh(row)

    # Live preview: run the brand-new (not-yet-deployed) candidate over the session the doctor is
    # reviewing, so they can visually compare its predictions against their own corrections before
    # deciding whether to approve it. This does NOT touch any Step row -- purely a response payload.
    preview_session_id = req.preview_session_id
    if preview_session_id is None:
        latest = (
            orm.query(db.RecordingSession)
            .order_by(db.RecordingSession.recording_date.desc(), db.RecordingSession.uploaded_at.desc())
            .first()
        )
        preview_session_id = latest.id if latest else None

    candidate_preview = None
    if preview_session_id is not None:
        preview_session_row = orm.get(db.RecordingSession, preview_session_id)
        if preview_session_row is not None:
            try:
                candidate_ml = mu.ModelVersion(row.version_dir)
                cand_steps, cand_preds = _predict_over_session_steps(orm, preview_session_id, candidate_ml)
                step_outs = []
                for s in cand_steps:
                    pred = cand_preds.get(s.id)
                    score, label = pred if pred is not None else (None, None)
                    step_outs.append(sch.StepOut(
                        timestamp_sec=s.timestamp_sec, spo2=s.spo2, bpm=s.bpm, motion_level=s.motion_level,
                        model_score=score, model_label=label, human_label=None,
                        effective_label=label, label_source="model",
                    ))
                candidate_preview = sch.TimelineResponse(
                    session=_session_summary(orm, preview_session_row), steps=step_outs,
                )
            except mu.MLError:
                candidate_preview = None  # non-fatal: comparison metrics still returned either way

    return sch.TrainResponse(
        model_version=_model_version_to_out(row),
        new_metrics=result["new_metrics"], parent_metrics=result["parent_metrics"],
        n_train_windows=int(len(X_w)) - result["n_test_windows"], n_test_windows=result["n_test_windows"],
        candidate_preview=candidate_preview,
    )


# --------------------------------------------------------------------------------------
# Model management + Phase 4: approval & firmware
# --------------------------------------------------------------------------------------

@app.get("/api/models", response_model=List[sch.ModelVersionOut])
def list_models(orm: OrmSession = Depends(get_db)):
    rows = orm.query(db.ModelVersion).order_by(db.ModelVersion.created_at.desc()).all()
    return [_model_version_to_out(r) for r in rows]


@app.post("/api/models/{version_id}/approve", response_model=sch.ApproveResponse)
def approve_model(version_id: int, orm: OrmSession = Depends(get_db)):
    row = orm.get(db.ModelVersion, version_id)
    if row is None:
        raise HTTPException(404, "Model version not found")
    if row.status != "pending_review":
        raise HTTPException(409, f"Model version status is '{row.status}', expected 'pending_review'.")

    active_row = _get_active_model_row(orm)
    if active_row is not None:
        # Snapshot what's about to be replaced into history BEFORE overwriting models/current/.
        # This is what preserves the very first, hand-provided model forever, and every generation
        # after it -- each superseded version gets exactly one permanent copy, taken at the moment
        # it stops being active.
        history_dir = MODELS_HISTORY_DIR / active_row.name
        history_dir.mkdir(parents=True, exist_ok=True)
        for fname in MODEL_FILES:
            src = MODELS_CURRENT_DIR / fname
            if src.exists():
                shutil.copy2(src, history_dir / fname)
        active_row.status = "archived"
        active_row.version_dir = str(history_dir)

    for fname in MODEL_FILES:
        src = Path(row.version_dir) / fname
        if src.exists():
            shutil.copy2(src, MODELS_CURRENT_DIR / fname)

    row.status = "active"
    row.version_dir = str(MODELS_CURRENT_DIR)
    orm.commit()

    # The candidate's files now live in models/current/ -- the staging copy is redundant, clear it
    # so nothing is ever duplicated between current/ and staging/.
    if MODELS_STAGING_DIR.exists():
        shutil.rmtree(MODELS_STAGING_DIR)
    MODELS_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    firmware_info = _build_firmware_for(row, orm)
    orm.refresh(row)
    return sch.ApproveResponse(model_version=_model_version_to_out(row), firmware=firmware_info)


def _build_firmware_for(row: db.ModelVersion, orm: OrmSession) -> dict:
    tflite_path = MODELS_CURRENT_DIR / "model_int8.tflite"
    if not tflite_path.exists():
        info = {"mode": "none", "success": False, "note": "No model_int8.tflite found to convert."}
        row.gbl_build_info_json = json.dumps(info)
        orm.commit()
        return info

    output_gbl = MODELS_CURRENT_DIR / "xG26_Devkit.gbl"
    result = gbl_bridge.convert_to_gbl(str(tflite_path), str(output_gbl), app_config_path=APP_CONFIG_PATH)
    row.gbl_path = result.get("gbl_path")
    row.gbl_build_info_json = json.dumps(result)
    orm.commit()
    return dict(result)


@app.post("/api/models/{version_id}/build-firmware", response_model=sch.BuildFirmwareResponse)
def build_firmware(version_id: int, orm: OrmSession = Depends(get_db)):
    row = orm.get(db.ModelVersion, version_id)
    if row is None:
        raise HTTPException(404, "Model version not found")
    if row.status != "active":
        raise HTTPException(409, "Only the active model version's firmware can be (re)built. Approve it first.")
    info = _build_firmware_for(row, orm)
    orm.refresh(row)
    return sch.BuildFirmwareResponse(model_version=_model_version_to_out(row), firmware=info)


# --------------------------------------------------------------------------------------
# Phase 4: OTA download endpoint for the Android app
# --------------------------------------------------------------------------------------

@app.get("/api/download/latest-model")
def download_latest_model(orm: OrmSession = Depends(get_db)):
    active_row = _get_active_model_row(orm)
    if active_row is None:
        raise HTTPException(404, "No active model version.")

    gbl_path = Path(active_row.gbl_path) if active_row.gbl_path else MODELS_CURRENT_DIR / "xG26_Devkit.gbl"
    if gbl_path.exists():
        return FileResponse(str(gbl_path), filename=gbl_path.name, media_type="application/octet-stream")

    tflite_path = MODELS_CURRENT_DIR / "model_int8.tflite"
    if tflite_path.exists():
        return FileResponse(
            str(tflite_path), filename=tflite_path.name, media_type="application/octet-stream",
        )
    raise HTTPException(404, "No .gbl or .tflite artifact available for the active model version.")


@app.get("/api/health")
def health():
    return {"status": "ok"}
