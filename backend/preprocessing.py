"""
preprocessing.py

Refactor of the user's `prepare_dataset.py` DSP pipeline into reusable functions.

IMPORTANT: The DSP math (SomniGuardDSP, SomniGuardMotion) below is copied VERBATIM
from prepare_dataset.py so that live-patient inference features are computed
identically to how the original training dataset was built. Do not tweak the DSP
constants here without also updating prepare_dataset.py, or the deployed model's
normalization stats will no longer match what this pipeline produces.

Key difference from prepare_dataset.py: that script's job was to build a *training*
set, so it required a ground-truth 'Label' column (breath-hold annotation) to derive
window labels via the 3-tier rule. In production, a patient's raw upload has no such
column (or if it does, e.g. from a research-mode recording, we deliberately ignore it
here) — labels are produced by the deployed model, then corrected by the doctor. So
everything below stops at "produce the 2Hz, 28-dim feature steps"; label derivation is
handled by ml_utils.py using the model + doctor's overrides instead of the 3-tier
raw-breath-hold heuristic.
"""

import csv
import math
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---- DSP constants (verbatim from prepare_dataset.py) ----
DSP_WINDOW_SIZE = 128
DSP_STRIDE = 25
BPM_FILTER_SIZE = 5
BPM_DEFAULT = 75
HPF_ALPHA = 0.97
LPF_BETA = 0.50
SPO2_A = 100.00
SPO2_B = -2.50
SPO2_C = 18.75
SPO2_SMOOTH = 0.40
SQI_MIN_PI = 0.001
SQI_MAX_PI = 0.200
MAX_SPO2_DROP = 2.50


def parse_float(row: dict, keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            try:
                return float(row[k])
            except ValueError:
                pass
    return default


def is_session_boundary(row: dict) -> bool:
    full_str = " ".join([str(v) for v in row.values() if v is not None])
    return ("---" in full_str or "NEW SESSION" in full_str or "CUT" in full_str)


class SomniGuardDSP:
    """Verbatim port of prepare_dataset.py's SomniGuardDSP (SpO2 / BPM estimator)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dc_track_red = 0.0
        self.dc_track_ir = 0.0
        self.lpf_red_prev = 0.0
        self.lpf_ir_prev = 0.0
        self.is_finger_attached = False
        self.bpm_state = 0
        self.valley_min_val = 0.0
        self.valley_min_time = 0
        self.last_beat_time = 0
        self.samples_since_last_beat = 0
        self.local_ac_ir_min = 0.0
        self.beat_threshold = -150.0
        self.bpm_index = 0
        self.smoothed_bpm = BPM_DEFAULT
        self.bpm_history = [BPM_DEFAULT] * BPM_FILTER_SIZE
        self.history_sq_red = [0.0] * DSP_WINDOW_SIZE
        self.history_sq_ir = [0.0] * DSP_WINDOW_SIZE
        self.history_dc_red = [0.0] * DSP_WINDOW_SIZE
        self.history_dc_ir = [0.0] * DSP_WINDOW_SIZE
        self.circular_index = 0
        self.sample_count = 0
        self.stride_counter = 0
        self.is_buffer_full = False
        self.final_r = 0.0
        self.final_spo2 = 98.0
        self.is_first_calc = True
        self.buf_spo2 = [98.0] * 5

    def _apply_hampel_filter(self, new_val: float) -> float:
        self.buf_spo2.pop(0)
        self.buf_spo2.append(new_val)
        sorted_x = sorted(self.buf_spo2)
        median_m = sorted_x[2]
        dev = sorted([abs(x - median_m) for x in self.buf_spo2])
        mad = dev[2]
        threshold = max(2.5, 3.0 * 1.4826 * mad)
        if abs(new_val - median_m) > threshold:
            return median_m
        return new_val

    def _update_bpm(self, ac_ir_filtered: float, timestamp_ms: int):
        self.samples_since_last_beat += 1
        if ac_ir_filtered < self.local_ac_ir_min:
            self.local_ac_ir_min = ac_ir_filtered

        if self.bpm_state == 0:
            if ac_ir_filtered < self.beat_threshold and self.samples_since_last_beat > 15:
                self.bpm_state = 1
                self.valley_min_val = ac_ir_filtered
                self.valley_min_time = timestamp_ms
        elif self.bpm_state == 1:
            if ac_ir_filtered < self.valley_min_val:
                self.valley_min_val = ac_ir_filtered
                self.valley_min_time = timestamp_ms
            if ac_ir_filtered > (self.valley_min_val + 25.0):
                delta_time = self.valley_min_time - self.last_beat_time
                if 375 < delta_time < 1500:
                    instant_bpm = 60000.0 / delta_time
                    self.bpm_history[self.bpm_index] = int(instant_bpm)
                    self.bpm_index = (self.bpm_index + 1) % BPM_FILTER_SIZE
                    self.smoothed_bpm = sum(self.bpm_history) // BPM_FILTER_SIZE
                self.last_beat_time = self.valley_min_time
                self.beat_threshold = self.valley_min_val * 0.60
                self.local_ac_ir_min = 0.0
                self.samples_since_last_beat = 0
                self.bpm_state = 0

        if self.samples_since_last_beat > 100:
            self.beat_threshold = max(-50.0, self.local_ac_ir_min * 0.5)
            self.local_ac_ir_min = 0.0
            self.samples_since_last_beat = 0
            self.last_beat_time = timestamp_ms
            self.bpm_state = 0

    def _calculate_spo2(self) -> Tuple[bool, float, float]:
        sum_sq_red = sum(self.history_sq_red)
        sum_sq_ir = sum(self.history_sq_ir)
        sum_dc_red = sum(self.history_dc_red)
        sum_dc_ir = sum(self.history_dc_ir)

        rms_red = math.sqrt(sum_sq_red / float(DSP_WINDOW_SIZE))
        rms_ir = math.sqrt(sum_sq_ir / float(DSP_WINDOW_SIZE))
        mean_dc_red = sum_dc_red / float(DSP_WINDOW_SIZE)
        mean_dc_ir = sum_dc_ir / float(DSP_WINDOW_SIZE)

        if rms_ir > 0.0 and mean_dc_red > 0.0 and mean_dc_ir > 0.0:
            pi_ir = rms_ir / mean_dc_ir
            pi_red = rms_red / mean_dc_red
            sqi_ok = (SQI_MIN_PI <= pi_ir <= SQI_MAX_PI) and (SQI_MIN_PI <= pi_red <= SQI_MAX_PI)

            instant_r = (rms_red / mean_dc_red) / (rms_ir / mean_dc_ir)
            instant_spo2 = SPO2_A - (SPO2_B * instant_r) - (SPO2_C * instant_r * instant_r)
            instant_spo2 = max(50.0, min(100.0, instant_spo2))

            if self.is_first_calc:
                self.final_spo2 = instant_spo2
                self.final_r = instant_r
                self.buf_spo2 = [instant_spo2] * 5
                self.is_first_calc = False
            elif sqi_ok:
                spo2_new = SPO2_SMOOTH * instant_spo2 + (1.0 - SPO2_SMOOTH) * self.final_spo2
                if spo2_new < self.final_spo2 - MAX_SPO2_DROP:
                    spo2_new = self.final_spo2 - MAX_SPO2_DROP
                self.final_spo2 = self._apply_hampel_filter(spo2_new)
                self.final_r = SPO2_SMOOTH * instant_r + (1.0 - SPO2_SMOOTH) * self.final_r

        return True, self.final_spo2, self.final_r

    def process_sample(self, raw_red: int, raw_ir: int, timestamp_ms: int) -> Tuple[bool, float, int, float, float]:
        red_f = float(raw_red)
        ir_f = float(raw_ir)

        if not self.is_finger_attached:
            self.dc_track_red = red_f
            self.dc_track_ir = ir_f
            self.lpf_red_prev = 0.0
            self.lpf_ir_prev = 0.0
            self.is_finger_attached = True
            self.is_first_calc = True
            self.last_beat_time = timestamp_ms
            self.samples_since_last_beat = 0
            return False, self.final_spo2, self.smoothed_bpm, 0.0, self.dc_track_ir

        ac_red_raw = red_f - self.dc_track_red
        self.dc_track_red = (1.0 - HPF_ALPHA) * red_f + HPF_ALPHA * self.dc_track_red
        ac_ir_raw = ir_f - self.dc_track_ir
        self.dc_track_ir = (1.0 - HPF_ALPHA) * ir_f + HPF_ALPHA * self.dc_track_ir

        ac_red_filtered = (1.0 - LPF_BETA) * ac_red_raw + LPF_BETA * self.lpf_red_prev
        self.lpf_red_prev = ac_red_filtered
        ac_ir_filtered = (1.0 - LPF_BETA) * ac_ir_raw + LPF_BETA * self.lpf_ir_prev
        self.lpf_ir_prev = ac_ir_filtered

        self._update_bpm(ac_ir_filtered, timestamp_ms)

        idx = self.circular_index
        self.history_sq_red[idx] = ac_red_filtered * ac_red_filtered
        self.history_sq_ir[idx] = ac_ir_filtered * ac_ir_filtered
        self.history_dc_red[idx] = self.dc_track_red
        self.history_dc_ir[idx] = self.dc_track_ir

        self.circular_index = (self.circular_index + 1) % DSP_WINDOW_SIZE
        self.sample_count += 1
        self.stride_counter += 1

        if not self.is_buffer_full and self.sample_count >= DSP_WINDOW_SIZE:
            self.is_buffer_full = True

        if self.is_buffer_full and self.stride_counter >= DSP_STRIDE:
            self.stride_counter = 0
            _, cur_spo2, _ = self._calculate_spo2()
            return True, cur_spo2, self.smoothed_bpm, ac_ir_filtered, self.dc_track_ir

        return False, self.final_spo2, self.smoothed_bpm, ac_ir_filtered, self.dc_track_ir


class SomniGuardMotion:
    """Verbatim port of prepare_dataset.py's SomniGuardMotion (motion_level estimator)."""

    def __init__(self, sample_rate_hz: int = 50, window_size: int = 50, threshold: float = 0.15):
        self.sample_rate_hz = sample_rate_hz
        self.window_size = window_size
        self.threshold = threshold
        self.ring_buffer = [0.0] * window_size
        self.head = 0
        self.count = 0
        self.sum_a = 0.0
        self.sum_sq_a = 0.0

    def process_sample(self, ax: float, ay: float, az: float) -> float:
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        if self.count < self.window_size:
            self.ring_buffer[self.head] = mag
            self.sum_a += mag
            self.sum_sq_a += mag * mag
            self.head = (self.head + 1) % self.window_size
            self.count += 1
        else:
            old_val = self.ring_buffer[self.head]
            self.sum_a += mag - old_val
            self.sum_sq_a += (mag * mag) - (old_val * old_val)
            self.ring_buffer[self.head] = mag
            self.head = (self.head + 1) % self.window_size

        if self.count < 2:
            return 0.0

        n = float(self.count)
        mean_a = self.sum_a / n
        var_a = (self.sum_sq_a / n) - (mean_a * mean_a)
        std_a = math.sqrt(max(0.0, var_a))
        return std_a


@dataclass
class StepFeature:
    """One 0.5s feature step: the 28-dim vector [spo2, bpm, 25x ir_norm, motion]."""
    segment_index: int
    step_index: int
    timestamp_sec: float
    spo2: float
    bpm: float
    ir_norm: List[float]  # length 25
    motion_level: float

    def as_vector(self) -> List[float]:
        return [self.spo2, float(self.bpm)] + list(self.ir_norm) + [self.motion_level]


class PreprocessingError(ValueError):
    """Raised for malformed/incompatible uploaded CSVs."""


def parse_recording_date(csv_filepath: str) -> Optional[dt.date]:
    """
    A whole CSV = one night's recording, so the "Date" column (format dd/mm/yy, e.g.
    "15/09/26") is expected to be constant across every row -- only the first row is
    read. Returns None if there's no Date column at all (older exports, before the
    mobile app started including it) or it can't be parsed; the caller decides the
    fallback (today's date, with a warning) in that case.
    """
    with open(csv_filepath, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        date_key = next((k for k in reader.fieldnames if k and k.strip().lower() == "date"), None)
        if date_key is None:
            return None
        for row in reader:
            raw = row.get(date_key)
            if raw is None or not str(raw).strip():
                continue
            raw = str(raw).strip()
            for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
                try:
                    return dt.datetime.strptime(raw, fmt).date()
                except ValueError:
                    continue
            return None  # a Date column exists but its value didn't match any known format
    return None


def parse_raw_segments(csv_filepath: str) -> List[List[dict]]:
    """
    Reads a raw CSV and splits it into continuous segments, exactly like
    prepare_dataset.py's parse_raw_sessions (renamed 'segments' here to avoid
    confusion with a DB-level upload 'Session', which may contain >1 segment).
    """
    segments: List[List[dict]] = []
    current_segment: List[dict] = []
    current_user = None

    with open(csv_filepath, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise PreprocessingError("CSV has no header row / could not be parsed.")

        for idx, row in enumerate(reader):
            full_str = " ".join([str(v) for v in row.values() if v is not None])
            if "---" in full_str or "NEW SESSION" in full_str or "CUT" in full_str:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue

            user_info = row.get("UserInfo", row.get("user", "")).strip() if row.get("UserInfo") or row.get("user") else None
            sample_id = parse_float(row, ["Sample ID", "sample_id", "id"], default=-1.0)

            if (user_info and current_user and user_info != current_user) or (sample_id == 0.0 and len(current_segment) > 0):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []

            current_user = user_info

            raw_red_val = parse_float(row, ["RED", "ppg_red", "red", "Red"], default=-1.0)
            raw_ir_val = parse_float(row, ["IR", "ppg_ir", "ir", "Ir"], default=-1.0)
            if raw_red_val < 0 or raw_ir_val < 0:
                continue

            current_segment.append(row)

    if current_segment:
        segments.append(current_segment)

    if not segments:
        raise PreprocessingError(
            "No valid rows found in CSV. Expected columns like Time/RED/IR/acc_x/acc_y/acc_z "
            "(see the sample export from the ring app)."
        )
    return segments


def compute_step_features(csv_filepath: str, ppg_sample_rate: int = 50) -> List[StepFeature]:
    """
    Runs the DSP pipeline over every segment in the CSV and returns the flat,
    chronologically-ordered list of 2Hz (0.5s-stride) feature steps.

    This intentionally stops BEFORE any windowing or labeling — that's handled by
    ml_utils.py, since window boundaries need to respect segment boundaries and
    labels now come from the model (then doctor corrections), not a raw 'Label'
    column.
    """
    segments = parse_raw_segments(csv_filepath)
    stride_raw_count = ppg_sample_rate // 2  # 25 raw samples per 0.5s stride

    all_steps: List[StepFeature] = []

    for seg_idx, segment_rows in enumerate(segments):
        dsp = SomniGuardDSP()
        motion = SomniGuardMotion(sample_rate_hz=ppg_sample_rate)

        stride_ac_ir_buffer: List[float] = []
        last_spo2 = 98.0
        last_bpm = 75.0
        last_ts_ms = 0
        step_idx_in_segment = 0

        for idx, row in enumerate(segment_rows):
            raw_red = int(parse_float(row, ["RED", "ppg_red", "red", "Red"], default=100000))
            raw_ir = int(parse_float(row, ["IR", "ppg_ir", "ir", "Ir"], default=100000))

            time_val = parse_float(row, ["Time", "timestamp_ms", "time", "timestamp_s"], default=idx * (1000.0 / ppg_sample_rate))
            ts_ms = int(time_val * 1000) if time_val < 10000 else int(time_val)
            last_ts_ms = ts_ms

            ax = parse_float(row, ["acc_x", "ax", "ACC_X", "AccX"], default=0.0)
            ay = parse_float(row, ["acc_y", "ay", "ACC_Y", "AccY"], default=0.0)
            az = parse_float(row, ["acc_z", "az", "ACC_Z", "AccZ"], default=1.0)

            if abs(ax) > 10.0 or abs(ay) > 10.0 or abs(az) > 10.0:
                ax /= 1000.0
                ay /= 1000.0
                az /= 1000.0

            has_new_stride, spo2, bpm, ac_ir, dc_ir = dsp.process_sample(raw_red, raw_ir, ts_ms)
            if has_new_stride:
                last_spo2 = spo2
                last_bpm = bpm

            ac_ir_norm = (ac_ir / dc_ir) if dc_ir > 0.0 else 0.0
            stride_ac_ir_buffer.append(ac_ir_norm)

            motion_val = motion.process_sample(ax, ay, az)

            if len(stride_ac_ir_buffer) >= stride_raw_count:
                all_steps.append(
                    StepFeature(
                        segment_index=seg_idx,
                        step_index=step_idx_in_segment,
                        timestamp_sec=last_ts_ms / 1000.0,
                        spo2=last_spo2,
                        bpm=float(last_bpm),
                        ir_norm=stride_ac_ir_buffer[:25],
                        motion_level=motion_val,
                    )
                )
                stride_ac_ir_buffer = []
                step_idx_in_segment += 1

    if not all_steps:
        raise PreprocessingError(
            "CSV parsed but produced zero 0.5s feature steps (recording too short, or all "
            "rows were filtered out as invalid)."
        )
    return all_steps
