# SomniGuard — AutoML & Human-in-the-Loop Platform (PoC)

A single-patient, multi-night doctor-in-the-loop platform. Every morning the mobile app
pushes the night's recording once it has Wi-Fi (for now, until that integration exists,
`scripts/ingest_recording.py` stands in for it); the doctor opens the dashboard, picks a
night, reviews the model's predictions on an interactive timeline, drag-corrects
mistakes, fine-tunes the on-device model from those corrections, and (optionally) builds
the resulting `.gbl` firmware image for OTA delivery to the watch.

This implements the 4 phases from the spec, matching your **actual** production ML
pipeline (from `SomniGuard_experiments_3.ipynb`) rather than a simplified generic
classifier — see "Design decisions" below for why, and what that means for you.

## What's inside

```
backend/
  main.py            FastAPI app — all /api/... endpoints
  database.py         SQLite schema (SQLAlchemy)
  preprocessing.py    DSP feature extraction (ported from your prepare_dataset.py)
  ml_utils.py          Compression, ordinal regression, threshold tuning, fine-tuning, quantization
  gbl_bridge.py         Wraps your build_gbl_pipeline.py, with a mock fallback
  schemas.py             Pydantic request/response models
  vendor/                Your original scripts, kept for reference (not executed directly)
frontend/
  app.py                Streamlit doctor dashboard (single linear flow, no sidebar)
config/
  app_config.json        Paths + defaults — edit this for your setup
models/
  current/                THE ACTIVE, DEPLOYED MODEL — put your real files here (see below)
  staging/                A fine-tuned candidate awaiting the doctor's approve decision (single slot)
  history/                Snapshots of models that USED to be active, one folder per past generation
scripts/
  make_dev_placeholder_model.py   OPTIONAL: fake model to test the app before yours is ready
  ingest_recording.py             Pushes a night's CSV into the platform (stands in for the
                                   future mobile-app-to-cloud push; not part of the doctor UI)
data/                       SQLite DB lives here at runtime — the ONLY persisted patient data;
                            raw uploaded CSVs are processed via a temp file and never kept
```

### Model lifecycle (current / staging / history)

- **`models/current/`** is always the single source of truth for "what's deployed." All
  inference (new uploads, timeline views) reads directly from here — there's no version
  picker anywhere in the UI; "current" always means whatever this folder holds.
- **`models/staging/`** holds one, and only one, fine-tuned candidate at a time, produced
  by `/api/train`. It's cleared and rewritten on every training run — training again
  before approving discards the previous candidate, so nothing accumulates.
- **`models/history/<name>/`** is written to only at the moment a candidate is approved:
  right before `models/current/` is overwritten, whatever's currently in there gets
  copied to `models/history/<its-own-name>/` first. This is what guarantees the very
  first model you provide is never lost, every past generation stays retrievable, and
  the currently-active files exist in exactly one place at any given time (no duplicate
  copies floating between `current/` and `history/`).

## 1. Setup

```bash
cd somniguard_platform
python3 -m venv .venv && source .venv/bin/activate   # or your usual venv approach
pip install -r requirements.txt
```

## 2. Plug in your real model

You said you'd provide these yourself — here's exactly what `models/current/` needs:

```
models/current/
  model_float.h5      A trainable Keras checkpoint (float weights)
  model_int8.tflite   The deployed, on-device int8 model
  config.json         Everything else (see schema below)
```

Once the three files are in place:

```bash
uvicorn main:app --app-dir backend --reload --port 8000     # from the project root

# in another terminal, once the server's up:
curl -X POST http://127.0.0.1:8000/api/models/register-initial
```

## 3. Get data into the platform

The doctor's dashboard has no upload button on purpose — in production, the mobile app
pushes each night's recording automatically once it has Wi-Fi, so by the time the doctor
opens the dashboard, the data's already there. Until that mobile integration exists, use
the same `/api/upload` endpoint directly via a small CLI script:

```bash
# !!! If folder data/ has files csv and db, we don't need to run these code anymore
python scripts/ingest_recording.py path/to/night1.csv path/to/night2.csv
# or against a non-default host:
python scripts/ingest_recording.py --base-url http://192.168.1.50:8000 night.csv
```

Each CSV is one night and needs a **`Date` column** (format `dd/mm/yy`, e.g. `15/09/26`,
constant across every row in the file) — that's what the doctor's day picker in the UI
filters on. If a CSV doesn't have one (e.g. today's sample export format, which
predates this field), the platform falls back to *today's date* and returns a warning;
it'll still work end-to-end, just without correct multi-night tracking. Add the `Date`
column to your mobile app's export format whenever that's convenient on your end — no
other change is needed here.

## 4. Run it

```bash
uvicorn main:app --app-dir backend --reload --port 8000

# Run on another terminal, once the server is up
streamlit run frontend/app.py
```

Then open the Streamlit URL it prints. The UI is one linear flow, top to bottom, no
sidebar, no upload button, and no model version picker:

1. **Select a night** — a date panel (pick from a calendar or type a date) showing
   whichever nights have been ingested so far. No upload control here — see "Get data
   into the platform" above for how data actually gets in.
2. **Model inference (active model)** — the timeline chart for that night; always
   reflects whichever model is currently active, even if you approved a new one since
   that night was ingested.
3. **Correct a time range** — drag-select (or type start/end minutes) + pick the right
   label. This edits that night's data in place; it never creates a new recording.
4. **Fine-tune & deploy** — trains from the active model's weights on every correction
   made so far (across every night, not just the one you're viewing), then shows the
   *new candidate's own* predictions on the night you're reviewing right there (so you
   can scroll up and compare its bands against what you just corrected) alongside
   accuracy/F1/confusion-matrix comparisons, and a deploy button.

## 5. Wire up real firmware builds (optional)

By default, `/api/models/{id}/build-firmware` (and the auto-build on approval) writes a
placeholder file and clearly marks the response `"mode": "mock"` — this sandbox/most dev
machines don't have the Silicon Labs SDK or Simplicity Commander installed, and your
`build_gbl_pipeline.py` is Windows-only.

To get real `.gbl` builds: run this whole app on your Windows dev machine (the one with
Simplicity Studio + Simplicity Commander installed), and set in `config/app_config.json`:

```json
"gbl_pipeline_script_path": "C:/path/to/your/actual/firmware/project/build_gbl_pipeline.py"
```

This must point at the **real** script sitting inside your actual firmware project — not
the copy vendored under `backend/vendor/` — because that script resolves its working
directories (`config/tflite/`, `cmake_gcc/build/`, `autogen/`) relative to its own
location. The vendored copy is there for reference only.

## 6. Android app OTA pull

`GET /api/download/latest-model` serves the current active model's `.gbl` (falling back
to `.tflite` if no `.gbl` exists yet). Point your Android app at
`http://<this-machine's-LAN-IP>:8000/api/download/latest-model`.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/models/register-initial` | Bootstrap: register `models/current/` as the active model (once) |
| POST | `/api/upload` | Ingest one night's raw CSV (not called from the doctor UI — see "Get data into the platform") |
| GET | `/api/sessions` | List ingested nights, each with its `recording_date` |
| GET | `/api/sessions/{id}/timeline` | Per-timestep data for the doctor's chart |
| POST | `/api/sessions/{id}/relabel` | Correct a time range (edits that night in place) |
| POST | `/api/train` | Fine-tune the active model on all corrected data |
| GET | `/api/models` | List model versions (active/pending/archived) |
| POST | `/api/models/{id}/approve` | Promote a pending version to active + build firmware |
| POST | `/api/models/{id}/build-firmware` | (Re)run the `.tflite` → `.gbl` conversion |
| GET | `/api/download/latest-model` | Serve the current `.gbl`/`.tflite` for OTA pull |

Interactive docs (Swagger) are auto-served at `http://127.0.0.1:8000/docs` once uvicorn
is running.

---

## Design decisions worth knowing about

- **Per-timestep storage, not per-window.** Windows overlap heavily (30s window, 0.5s
  stride → 98% overlap between consecutive windows), so the DB stores one row per 0.5s
  feature step. Windows are reconstructed on the fly for training/inference by slicing
  60 consecutive steps. This is also the *only* thing persisted from an upload — the
  raw CSV itself is processed via a temp file and deleted immediately, per the "don't
  hoard dataset files" requirement. If you need to re-derive raw signals later, you'd
  need to re-upload the original recording; only the corrected/predicted labels and
  derived features survive.
- **Window labels via majority vote, tie-broken toward the more severe class.** When
  reconstructing a training window from per-step labels (doctor overrides + model
  predictions), ties are broken toward the more severe class deliberately, since a
  missed apnea event is worse than a false alarm.
- **Fine-tuning reuses the parent model's normalization stats (mean/std), not
  recomputed ones.** This keeps the existing float weights' learned scale meaningful —
  only the weights adapt to the new patient data, which matches "do NOT freeze base
  layers" while still being a *gentle* adaptation.
- **Fine-tuning uses a proper 3-way split** (train / threshold-tuning / test) under the
  hood, not just a single 80/20 split — this matches your notebook's actual
  methodology (EarlyStopping needs its own held-out loss signal, and tau_1/tau_2
  retuning needs data the model never trained on). The system prompt's literal "20%
  validation split" wording is satisfied by the tuning+test holdout combined; this is a
  guardrail applied silently, per "the doctor doesn't have to manage hyperparameters."
- **Every timeline view live-recomputes predictions against whichever model is
  CURRENTLY active** (and caches the result back onto the Step rows) rather than
  trusting whatever was computed at upload time. So if you approve a new model, then
  scroll back to an older recording, you'll see that new model's opinion of it — not a
  stale prediction from whatever was active when it was uploaded. Training's "model
  label when the doctor hasn't corrected a step" falls back to that cache, which in
  practice is fresh because viewing a session's timeline is always the step right
  before correcting/training it in the linear UI flow.
- **Causal windowing, so the first ~30s of every recording has no prediction.** A
  window needs 60 steps of trailing context; the first 59 steps of each segment show up
  as unlabeled on the timeline. This is expected, not a bug.
- **Model comparison (new vs. active) is evaluated on the same fresh test split**, so
  it's a fair apples-to-apples comparison rather than comparing against the parent's
  historical (differently-split) metrics.
- **The candidate preview** (section 4) runs the brand-new, not-yet-approved model over
  the recording you're currently reviewing and colors its timeline by the candidate's
  *own* predicted labels (not your corrections) — specifically so you can scroll up and
  visually check whether it now agrees with what you corrected, rather than only
  trusting the aggregate accuracy/F1 numbers.
- **The relabel brush-select** uses Streamlit's `on_select="rerun"` on `st.plotly_chart`
  (Streamlit ≥ 1.35). If your installed Streamlit version handles that differently, the
  numeric start/end (minutes) inputs below the chart are the reliable fallback and
  always work regardless.
- **No auth, single implicit patient, SQLite, no session picker.** Matches the PoC
  scope in the spec — the UI navigates by date (a "day panel"), not a recording-ID
  picker; this is not designed to be internet-facing or multi-tenant as-is.
  `GET /api/models` and `GET /api/sessions` still exist as plain API endpoints (useful
  via `/docs` or `curl` for debugging/auditing) even though the UI doesn't surface a
  version list or session picker.
- **One CSV = one night**, matched by its `Date` column (`dd/mm/yy`), not by when it
  happened to be ingested. If two recordings land on the same date, the most recent
  ingestion wins for that day's display (`uploaded_at desc` tiebreak) — this shouldn't
  come up in normal use (one ring session per night) but is handled predictably rather
  than crashing.

## What I verified before handing this off

I don't have your real model weights, so I generated a placeholder model (same
architecture/config shape, random weights — see `scripts/make_dev_placeholder_model.py`)
and ran the entire flow end-to-end against it, including a specific test of the
current/staging/history lifecycle across **three consecutive generations**: register
initial → upload → relabel → fine-tune → approve (checksummed to confirm the initial
model landed byte-identical in `models/history/`) → upload again → relabel → fine-tune
→ approve again (checksummed to confirm generation 2 *also* archived correctly, with
generation 1 still untouched) → confirmed `models/staging/` is empty after every
approval and never accumulates across repeated un-approved training runs. Separately, I
generated 3 synthetic nights with a `Date` column (`dd/mm/yy`) plus the older
no-`Date`-column format, ingested all 4 via `scripts/ingest_recording.py`, and confirmed:
dates parse and store correctly, the no-`Date` CSV falls back to today's date with a
visible warning, `GET /api/sessions` sorts by recording date (not upload order), the
day-picker's date→session lookup logic behaves correctly including for a date with no
data, and a full relabel→fine-tune→approve cycle on a *dated* night leaves exactly one
session row for that date (no duplicate dataset created by correcting labels). I also
ran the mock and the *real* firmware-build subprocess hook (with a stand-in script,
since I don't have Windows/Simplicity Commander here), and confirmed no raw CSV ever
survives an upload on disk. Along the way I caught and fixed two real bugs in the
Streamlit dashboard: pandas turning the mixed int/null label column into `NaN` (which
would have crashed on a `dict[nan]` lookup), and Plotly silently dropping `row="all"`
shapes when `add_vrect` is called before any traces exist.

What I have **not** been able to test: your real model's actual numbers (obviously),
and the real Windows/Simplicity-Commander firmware path. Also worth a light
read-through: the CSV column-name matching in `preprocessing.py`
(`RED`/`IR`/`acc_x` etc. with a few fallback aliases) — I matched it to your sample
export and `prepare_dataset.py`, but if your mobile app's real export uses different
column names, that's a one-line fix in `parse_float(row, [...])`'s alias lists.
