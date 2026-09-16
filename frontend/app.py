"""
app.py - Streamlit doctor dashboard for the SomniGuard platform.

Deliberately minimal, single-column flow (no sidebar, no upload button, no model
version picker). The platform is a passive receiver of nightly data -- the mobile app
pushes each night's recording once it has Wi-Fi in the morning (for now, until that
integration exists, use scripts/ingest_recording.py to push data in) -- so by the time
the doctor opens this dashboard, the data is already there. The doctor:

    1. Picks a night to review (date panel)
    2. Sees that night's active-model inference
    3. Corrects a time range
    4. Fine-tunes & deploys -> shows the CANDIDATE's own inference (for visual
       comparison against the corrections above) + performance metrics, then a
       deploy decision

Talks to the FastAPI backend over HTTP (see backend/main.py). Run both:
    uvicorn main:app --app-dir backend --reload --port 8000
    streamlit run frontend/app.py
(See README.md for the exact commands / working directory.)
"""

import json
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "app_config.json"
with open(APP_CONFIG_PATH) as f:
    APP_CFG = json.load(f)
BASE_URL = APP_CFG.get("backend_base_url", "http://127.0.0.1:8000")

LABEL_NAMES = {0: "Normal", 1: "Abnormal", 2: "Sleep Apnea"}
LABEL_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}

st.set_page_config(page_title="SomniGuard Doctor Dashboard", layout="wide")


# --------------------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------------------

def api_get(path, **kwargs):
    r = requests.get(f"{BASE_URL}{path}", **kwargs)
    if not r.ok:
        st.error(f"GET {path} failed ({r.status_code}): {r.text}")
        r.raise_for_status()
    return r.json()


def api_post(path, json_body=None, **kwargs):
    r = requests.post(f"{BASE_URL}{path}", json=json_body, **kwargs)
    if not r.ok:
        st.error(f"POST {path} failed ({r.status_code}): {r.text}")
        r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------------------
# Shared: build the 3-row timeline figure (SpO2 / BPM / severity score + color bands).
# Used both for the active model's view and the fine-tuned candidate's preview, so the
# doctor can visually compare them using the exact same chart layout.
# --------------------------------------------------------------------------------------

def steps_to_df(steps: list) -> pd.DataFrame:
    df = pd.DataFrame(steps)
    df["minutes"] = df["timestamp_sec"] / 60.0
    return df


def label_runs(df: pd.DataFrame):
    """Contiguous runs of the same effective_label. NaN-safe: pandas turns the mixed
    int/null label column into float64 (NaN for missing), and NaN != NaN is always True,
    so a naive equality check would fragment every unlabeled stretch into single-row runs
    and later crash on a dict[nan] lookup. Normalize to plain int-or-None first."""
    runs = []
    cur_label = "__unset__"
    run_start = None
    prev_minute = None
    for _, row in df.iterrows():
        raw = row["effective_label"]
        lbl = None if pd.isna(raw) else int(raw)
        if lbl != cur_label:
            if cur_label not in (None, "__unset__") and run_start is not None:
                runs.append((run_start, prev_minute, cur_label))
            run_start = row["minutes"]
            cur_label = lbl
        prev_minute = row["minutes"]
    if cur_label not in (None, "__unset__") and run_start is not None:
        runs.append((run_start, prev_minute, cur_label))
    return runs


def build_timeline_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("SpO2 (%)", "Heart rate (BPM)", "Severity score"),
        row_heights=[0.35, 0.35, 0.3],
    )
    # Traces must be added BEFORE add_vrect(row="all", ...) -- add_vrect needs the subplot
    # axes to already exist to resolve "all"; called first, it silently adds zero shapes.
    fig.add_trace(go.Scatter(x=df["minutes"], y=df["spo2"], mode="lines", name="SpO2",
                              line=dict(color="#2980b9")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["minutes"], y=df["bpm"], mode="lines", name="BPM",
                              line=dict(color="#8e44ad")), row=2, col=1)
    score_df = df.dropna(subset=["model_score"])
    fig.add_trace(go.Scatter(x=score_df["minutes"], y=score_df["model_score"], mode="lines",
                              name="Severity score", line=dict(color="#34495e")), row=3, col=1)

    for start, end, lbl in label_runs(df):
        fig.add_vrect(
            x0=start, x1=end, fillcolor=LABEL_COLORS[lbl], opacity=0.18,
            layer="below", line_width=0, row="all", col=1,
        )

    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40))
    return fig


def label_count_row(df: pd.DataFrame):
    counts = df["effective_label"].value_counts(dropna=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total steps", len(df))
    c2.metric("Normal", int(counts.get(0, 0)))
    c3.metric("Abnormal", int(counts.get(1, 0)))
    c4.metric("Sleep Apnea", int(counts.get(2, 0)))


# --------------------------------------------------------------------------------------
# Connection + bootstrap check (quiet -- only visible when something needs attention)
# --------------------------------------------------------------------------------------

st.title("SomniGuard Doctor Dashboard")

try:
    api_get("/api/health")
except Exception:
    st.error(f"Cannot reach backend at {BASE_URL}. Is uvicorn running?")
    st.stop()

models = api_get("/api/models")
active_model = next((m for m in models if m["status"] == "active"), None)

if active_model is None:
    st.warning(
        "No active model yet. Place model_float.h5 / model_int8.tflite / config.json in "
        "models/current/, then register it below."
    )
    if st.button("Register models/current/ as active model"):
        try:
            resp = api_post("/api/models/register-initial")
            st.success(f"Registered '{resp['name']}' as active.")
            st.rerun()
        except Exception:
            pass
    st.stop()


# --------------------------------------------------------------------------------------
# 1. Select a night to review. There is no upload control here on purpose -- the
# platform receives data from the mobile app (or, for now, scripts/ingest_recording.py)
# before the doctor ever opens this dashboard.
# --------------------------------------------------------------------------------------

st.header("1. Select a night")

sessions = api_get("/api/sessions")
if not sessions:
    st.info(
        "No recordings in the platform yet. Data arrives automatically from the mobile "
        "app each morning; for now, push a CSV in with:\n\n"
        "`python scripts/ingest_recording.py path/to/night.csv`"
    )
    st.stop()

sessions_by_date = {}
for s in sessions:
    d = dt.date.fromisoformat(s["recording_date"])
    # If more than one recording landed on the same date, keep the most recent (sessions
    # is already ordered recording_date desc, uploaded_at desc, so the first one wins).
    sessions_by_date.setdefault(d, s)

available_dates = sorted(sessions_by_date.keys(), reverse=True)
selected_date = st.date_input("Night (you can also type a date directly)", value=available_dates[0])
st.caption(
    "Nights with data: " + ", ".join(d.isoformat() for d in available_dates)
    if len(available_dates) <= 15 else
    f"{len(available_dates)} nights available, most recent: {available_dates[0].isoformat()}"
)

session_summary = sessions_by_date.get(selected_date)
if session_summary is None:
    st.info(f"No recording for {selected_date.isoformat()}. Pick another date above.")
    st.stop()

session_id = session_summary["id"]

st.divider()

# --------------------------------------------------------------------------------------
# 2. Visualization of the active model's inference
# --------------------------------------------------------------------------------------

st.header(f"2. Model inference (active model) — {selected_date.isoformat()}")

timeline = api_get(f"/api/sessions/{session_id}/timeline")
for w in timeline.get("warnings", []):
    st.warning(w)
steps = timeline["steps"]
if not steps:
    st.warning("This recording has no steps.")
    st.stop()

df = steps_to_df(steps)
label_count_row(df)
st.caption(
    "Shaded bands: green = Normal, yellow = Abnormal, red = Sleep Apnea "
    "(active model's prediction, or your correction where applied)."
)
fig = build_timeline_figure(df)
selection_event = st.plotly_chart(
    fig, use_container_width=True, key="current_timeline_chart",
    on_select="rerun", selection_mode=["box"],
)

# Try to read the brush-selected x-range (Streamlit >= 1.35). The numeric inputs below
# always work regardless, as a reliable fallback across Streamlit versions.
brushed_start_min, brushed_end_min = None, None
try:
    boxes = selection_event["selection"]["box"]
    if boxes:
        xr = boxes[0]["x"]
        brushed_start_min, brushed_end_min = min(xr), max(xr)
except Exception:
    pass

st.divider()

# --------------------------------------------------------------------------------------
# 3. Correct a time range -- edits THIS night's data in place; never creates a new
# recording/session. A correction is just a label change on existing steps.
# --------------------------------------------------------------------------------------

st.header("3. Correct a time range")
r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
default_start = brushed_start_min if brushed_start_min is not None else float(df["minutes"].min())
default_end = brushed_end_min if brushed_end_min is not None else float(df["minutes"].min())
start_min = r1.number_input("Start (min)", min_value=float(df["minutes"].min()),
                             max_value=float(df["minutes"].max()), value=default_start, step=0.5)
end_min = r2.number_input("End (min)", min_value=float(df["minutes"].min()),
                           max_value=float(df["minutes"].max()), value=default_end, step=0.5)
new_label = r3.radio("Correct label", options=[0, 1, 2], format_func=lambda v: LABEL_NAMES[v], horizontal=True)
if r4.button("Apply correction", type="primary"):
    try:
        resp = api_post(
            f"/api/sessions/{session_id}/relabel",
            {"start_time": start_min * 60.0, "end_time": end_min * 60.0, "new_label": new_label},
        )
        st.success(f"Updated {resp['n_steps_affected']} steps.")
        st.rerun()
    except Exception:
        pass

st.divider()

# --------------------------------------------------------------------------------------
# 4. Fine-tune & deploy: train, preview the candidate's own inference right here for
# comparison against the corrections above, show performance, then let the doctor decide.
# --------------------------------------------------------------------------------------

st.header("4. Fine-tune & deploy")
st.caption("Trains from the active model's weights on ALL corrected nights so far.")

if st.button("Train new model", type="primary"):
    with st.spinner("Fine-tuning (compression -> ordinal regression -> threshold re-tuning -> int8 quantization)..."):
        try:
            result = api_post("/api/train", {"preview_session_id": session_id})
            st.session_state["train_result"] = result
        except Exception:
            pass

if "train_result" in st.session_state:
    result = st.session_state["train_result"]
    new_m, parent_m = result["new_metrics"], result["parent_metrics"]

    if result.get("candidate_preview"):
        st.subheader(f"Candidate model's inference — {selected_date.isoformat()} (compare against your corrections above)")
        cand_df = steps_to_df(result["candidate_preview"]["steps"])
        label_count_row(cand_df)
        st.caption(
            "Shaded bands here show the NEW candidate's OWN predictions (not your corrections) -- "
            "scroll up to compare against what you set in section 2/3."
        )
        st.plotly_chart(build_timeline_figure(cand_df), use_container_width=True, key="candidate_timeline_chart")
    else:
        st.info("Candidate preview unavailable for this recording (non-fatal -- metrics below are still valid).")

    st.subheader("Performance: candidate vs. currently active model")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Accuracy (new)", f"{new_m['accuracy']:.3f}", delta=f"{new_m['accuracy'] - parent_m['accuracy']:+.3f}")
    mcol2.metric("F1 macro (new)", f"{new_m['f1_macro']:.3f}", delta=f"{new_m['f1_macro'] - parent_m['f1_macro']:+.3f}")
    ap_new = new_m["per_class"]["Emergency"]["f1"]
    ap_parent = parent_m["per_class"]["Emergency"]["f1"]
    mcol3.metric("Sleep-Apnea F1 (new)", f"{ap_new:.3f}", delta=f"{ap_new - ap_parent:+.3f}")

    cm_col1, cm_col2 = st.columns(2)
    for col, m, title in ((cm_col1, parent_m, "Active model (test split)"), (cm_col2, new_m, "Candidate (same test split)")):
        cm = m["confusion_matrix"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred Normal", "Pred Abnormal", "Pred Apnea"],
            y=["True Normal", "True Abnormal", "True Apnea"],
            colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}",
        ))
        fig_cm.update_layout(title=title, height=350, margin=dict(t=40, b=10))
        col.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        f"Trained on {result['n_train_windows']} windows, evaluated on {result['n_test_windows']} "
        f"held-out windows. tau_1={new_m['tau_1']:.3f}, tau_2={new_m['tau_2']:.3f} "
        f"(re-tuned on a separate tuning split, never seen during training or final evaluation)."
    )

    if st.button("Approve & deploy this model", type="primary"):
        with st.spinner("Deploying + building firmware..."):
            try:
                approve_resp = api_post(f"/api/models/{result['model_version']['id']}/approve")
                fw = approve_resp["firmware"]
                if fw.get("mode") == "real" and fw.get("success"):
                    st.success("Model approved and deployed. Real .gbl firmware built successfully.")
                elif fw.get("mode") == "mock":
                    st.warning(f"Model approved and deployed. Firmware build was MOCKED: {fw.get('note')}")
                else:
                    st.error(f"Model approved and deployed, but firmware build failed: {fw.get('note')}")
                st.caption(f"Android app OTA URL: `{BASE_URL}/api/download/latest-model`")
                del st.session_state["train_result"]
                st.rerun()
            except Exception:
                pass
