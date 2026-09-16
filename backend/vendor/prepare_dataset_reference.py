#!/usr/bin/env python3
"""
prepare_dataset.py

Script bóc tách dữ liệu chuỗi thời gian 50Hz từ dataset.csv thành các cửa sổ trượt 30s
(stride 0.5s, tần số cập nhật 2Hz - 60 bước x 28 đặc trưng) cho dự án SomniGuard (IoT Challenge 2026).

Input Tensor X Shape: (N, 60, 28) gồm:
1. Cột [0]:     SpO2_est: Chỉ số bão hòa Oxy ước tính (%)
2. Cột [1]:     BPM_est: Nhịp tim ước tính (BPM)
3. Cột [2..26]: 25 mẫu PPG_AC_Norm_IR (AC_IR / DC_IR) trong 0.5s @ 50Hz
4. Cột [27]:    motion_level: Độ lệch chuẩn gia tốc cựa tay (g)

Output Label Y Shape: (N,) gán theo 3-Tier Physiological Severity Strategy:
- Nhãn 0 (Normal): R_total < 0.10 VÀ R_tail10 < 0.20
- Nhãn 1 (Mild / Onset Apnea): Đang nín thở (R_tail10 >= 0.50 HOẶC R_total >= 0.40) nhưng chưa kéo dài (D_tail < 40s VÀ SpO2 >= 93%)
- Nhãn 2 (Severe / Critical Apnea): Đang nín thở kéo dài nguy kịch (D_tail >= 40s HOẶC (D_tail >= 25s VÀ SpO2 < 93%))
- Discard (-1): Loại bỏ các cửa sổ ranh giới nhiễu chuyển tiếp

Xuất file nén: somniguard_train_dataset.npz chứa X (float32) và Y (int64)
"""

import sys
import os
import csv
import math
from typing import List, Dict, Tuple
import numpy as np

# Import SomniGuard DSP & Motion từ process_ppg_imu.py nếu có
try:
    from process_ppg_imu import SomniGuardDSP, SomniGuardMotion, parse_float, is_session_boundary, DSP_WINDOW_SIZE
except ImportError:
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


def parse_raw_sessions(csv_filepath: str) -> List[List[dict]]:
    """
    Đọc dataset.csv và chia thành các danh sách mẫu liên tục cho từng Session.
    """
    sessions: List[List[dict]] = []
    current_session: List[dict] = []
    current_user = None

    print(f"Đang đọc và phân đoạn session từ {csv_filepath}...")
    with open(csv_filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            full_str = " ".join([str(v) for v in row.values() if v is not None])
            if "---" in full_str or "NEW SESSION" in full_str or "CUT" in full_str:
                if current_session:
                    sessions.append(current_session)
                    current_session = []
                continue

            user_info = row.get('UserInfo', row.get('user', '')).strip() if row.get('UserInfo') or row.get('user') else None
            sample_id = parse_float(row, ['Sample ID', 'sample_id', 'id'], default=-1.0)

            if (user_info and current_user and user_info != current_user) or (sample_id == 0.0 and len(current_session) > 0):
                if current_session:
                    sessions.append(current_session)
                    current_session = []

            current_user = user_info

            raw_red_val = parse_float(row, ['RED', 'ppg_red', 'red', 'Red'], default=-1.0)
            raw_ir_val = parse_float(row, ['IR', 'ppg_ir', 'ir', 'Ir'], default=-1.0)
            if raw_red_val < 0 or raw_ir_val < 0:
                continue

            current_session.append(row)

    if current_session:
        sessions.append(current_session)

    print(f"Đã phân tách được {len(sessions)} phiên làm việc (sessions) liên tục từ dataset.")
    return sessions


def build_npz_dataset(csv_filepath: str, output_npz_path: str = "somniguard_train_dataset.npz", ppg_sample_rate: int = 50):
    """
    Xử lý các session, tạo cửa sổ trượt 30s (60 bước x 28 đặc trưng), áp dụng chiến lược gán nhãn 3 lớp (0, 1, 2)
    và xuất file npz chuẩn (N, 60, 28).
    """
    sessions = parse_raw_sessions(csv_filepath)

    X_list: List[np.ndarray] = []
    Y_list: List[int] = []

    total_candidate_windows = 0
    kept_label_0 = 0
    kept_label_1 = 0
    kept_label_2 = 0
    discarded_windows = 0

    stride_raw_count = ppg_sample_rate // 2         # 25 mẫu thô cho 0.5s stride
    window_30s_raw_count = ppg_sample_rate * 30    # 1500 mẫu thô cho 30s
    window_steps = 60                               # 60 bước 0.5s trong 30s

    for sess_idx, session_rows in enumerate(sessions):
        dsp = SomniGuardDSP()
        motion = SomniGuardMotion(sample_rate_hz=ppg_sample_rate)

        session_feature_steps: List[List[float]] = []
        session_raw_labels: List[int] = []
        continuous_bh_samples: List[int] = []
        current_bh_count = 0

        stride_ac_ir_buffer: List[float] = []
        last_spo2 = 98.0
        last_bpm = 75.0
        last_motion = 0.0

        # 1. Trích xuất chuỗi 28 đặc trưng (2Hz), nhãn thô và bộ đếm thời gian nín thở liên tục
        for idx, row in enumerate(session_rows):
            raw_red = int(parse_float(row, ['RED', 'ppg_red', 'red', 'Red'], default=100000))
            raw_ir = int(parse_float(row, ['IR', 'ppg_ir', 'ir', 'Ir'], default=100000))

            time_val = parse_float(row, ['Time', 'timestamp_ms', 'time', 'timestamp_s'], default=idx * (1000.0 / ppg_sample_rate))
            ts_ms = int(time_val * 1000) if time_val < 10000 else int(time_val)

            ax = parse_float(row, ['acc_x', 'ax', 'ACC_X', 'AccX'], default=0.0)
            ay = parse_float(row, ['acc_y', 'ay', 'ACC_Y', 'AccY'], default=0.0)
            az = parse_float(row, ['acc_z', 'az', 'ACC_Z', 'AccZ'], default=1.0)

            if abs(ax) > 10.0 or abs(ay) > 10.0 or abs(az) > 10.0:
                ax /= 1000.0
                ay /= 1000.0
                az /= 1000.0

            raw_label = parse_float(row, ['Label', 'label', 'LABEL'], default=0.0)
            is_bh = 1 if raw_label >= 1.0 else 0
            session_raw_labels.append(is_bh)

            # Đếm số mẫu nín thở liên tục tích lũy
            if is_bh == 1:
                current_bh_count += 1
            else:
                current_bh_count = 0
            continuous_bh_samples.append(current_bh_count)

            has_new_stride, spo2, bpm, ac_ir, dc_ir = dsp.process_sample(raw_red, raw_ir, ts_ms)
            if has_new_stride:
                last_spo2 = spo2
                last_bpm = bpm

            # Chuẩn hóa AC_IR (AC_IR / DC_IR) cho từng mẫu trong chu kỳ 25 mẫu (0.5s)
            ac_ir_norm = (ac_ir / dc_ir) if dc_ir > 0.0 else 0.0
            stride_ac_ir_buffer.append(ac_ir_norm)

            motion_val = motion.process_sample(ax, ay, az)
            last_motion = motion_val

            # Khi gom đủ 25 mẫu thô (1 stride = 0.5s) -> tạo vector 28 đặc trưng
            if len(stride_ac_ir_buffer) >= stride_raw_count:
                # 28 Đặc trưng: [SpO2 (1), BPM (1), 25 x PPG_AC_Norm_IR (25), motion_level (1)]
                feature_vec = [last_spo2, float(last_bpm)] + stride_ac_ir_buffer[:25] + [last_motion]
                session_feature_steps.append(feature_vec)
                stride_ac_ir_buffer = []

        # 2. Tạo cửa sổ trượt 30s (60 bước = 1500 mẫu thô) trên Session
        num_steps = len(session_feature_steps)
        if num_steps < window_steps:
            continue

        for i in range(num_steps - window_steps + 1):
            total_candidate_windows += 1

            # Ma trận 60 bước x 28 đặc trưng
            feat_matrix = np.array(session_feature_steps[i : i + window_steps], dtype=np.float32)

            # Lấy 1500 nhãn thô tương ứng với cửa sổ 30s
            raw_start_idx = i * stride_raw_count
            raw_end_idx = raw_start_idx + window_30s_raw_count
            window_raw_labels = session_raw_labels[raw_start_idx : raw_end_idx]

            if len(window_raw_labels) < window_30s_raw_count:
                continue

            # Thời gian nín thở liên tục tính đến thời điểm cuối cửa sổ (giây)
            tail_bh_duration_sec = continuous_bh_samples[raw_end_idx - 1] / float(ppg_sample_rate)

            # Giá trị SpO2 ở bước cuối cùng của cửa sổ (Cột 0)
            last_window_spo2 = feat_matrix[-1, 0]

            # Tính R_total (1500 mẫu = 30s) và R_tail10 (500 mẫu cuối = 10s)
            r_total = sum(window_raw_labels) / float(window_30s_raw_count)
            r_tail10 = sum(window_raw_labels[-500:]) / 500.0

            # Áp dụng chiến lược gán nhãn 3 lớp (0: Normal, 1: Mild Apnea, 2: Severe Apnea)
            assigned_label = -1

            if r_total < 0.10 and r_tail10 < 0.20:
                # Nhãn 0: Bình thường / Thở ổn định
                assigned_label = 0
                kept_label_0 += 1
            elif r_tail10 >= 0.50 or r_total >= 0.40:
                # Đang trong pha nín thở -> Phân cấp mức độ nguy kịch
                # Tiêu chí Nhãn 2 (Severe): Nín thở >= 40s HOẶC (nín thở >= 25s và SpO2 < 93.0%)
                if tail_bh_duration_sec >= 40.0 or (tail_bh_duration_sec >= 25.0 and last_window_spo2 < 93.0):
                    assigned_label = 2  # SEVERE / CRITICAL APNEA
                    kept_label_2 += 1
                else:
                    assigned_label = 1  # MILD / ONSET APNEA
                    kept_label_1 += 1
            else:
                # Discard cửa sổ ranh giới nhiễu chuyển tiếp
                discarded_windows += 1
                continue

            X_list.append(feat_matrix)
            Y_list.append(assigned_label)

    if not X_list:
        print("Lỗi: Không tạo được cửa sổ hợp lệ nào từ dataset!")
        return

    # 3. Chuyển đổi thành NumPy array chuẩn (N, 60, 28) và (N,)
    X = np.array(X_list, dtype=np.float32)  # Shape (N, 60, 28)
    Y = np.array(Y_list, dtype=np.int64)    # Shape (N,)

    # 4. Xuất file .npz nén
    np.savez_compressed(output_npz_path, X=X, Y=Y)
    print(f"\n=======================================================")
    print(f"  SOMNIGUARD AI DATASET (60x28) GENERATED SUCCESSFULLY")
    print(f"=======================================================")
    print(f"File đầu ra:        {output_npz_path}")
    print(f"Dung lượng file:     {os.path.getsize(output_npz_path) / (1024 * 1024):.2f} MB")
    print(f"Hình dạng Tensor X:  {X.shape} (dtype={X.dtype}) [60 bước x 28 đặc trưng]")
    print(f"Hình dạng Nhãn Y:   {Y.shape} (dtype={Y.dtype}) [3 lớp: 0, 1, 2]")
    print(f"-------------------------------------------------------")
    print(f"THỐNG KÊ CHI TIẾT CÁC CỬA SỔ (CLASS BALANCE - 3 LỚP):")
    print(f"Tổng cửa sổ trượt 30s đánh giá: {total_candidate_windows:,}")
    print(f"Tổng cửa sổ được lưu giữ (N):  {len(Y_list):,} ({(len(Y_list)/total_candidate_windows)*100:.2f}%)")
    print(f" - Nhãn 0 (Normal):             {kept_label_0:,} ({(kept_label_0/len(Y_list))*100:.2f}% của tập train)")
    print(f" - Nhãn 1 (Mild / Onset Apnea): {kept_label_1:,} ({(kept_label_1/len(Y_list))*100:.2f}% của tập train)")
    print(f" - Nhãn 2 (Severe Apnea / Hyp): {kept_label_2:,} ({(kept_label_2/len(Y_list))*100:.2f}% của tập train)")
    print(f"Số cửa sổ ranh giới bị loại:    {discarded_windows:,} ({(discarded_windows/total_candidate_windows)*100:.2f}% tổng số)")
    print(f"=======================================================\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    input_file = "dataset.csv"
    output_npz = "somniguard_train_dataset.npz"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_npz = sys.argv[2]

    build_npz_dataset(input_file, output_npz, ppg_sample_rate=50)
