"""
make_dev_placeholder_model.py

OPTIONAL, dev-only convenience. You said you'll plug in your own real
model_float.h5 / model_int8.tflite / config.json (matching your notebook's export
format) into models/current/ -- this script is NOT required for that.

Use this only if you want to click through the app's full flow (upload -> dashboard
-> relabel -> train -> approve -> firmware) before your real model files are ready,
to confirm the plumbing works. The resulting model is trained on random noise and
has zero real predictive value -- replace models/current/ with your real files
before doing anything clinical.

Run from the project root:
    python scripts/make_dev_placeholder_model.py
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import ml_utils as mu  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "current"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["mean", "std", "rms", "range"]
N_CHANNELS = len(FEATURES) + 3
WINDOW_FRAMES = 60

CH_CFG = {
    "window_frames": WINDOW_FRAMES, "frame_samples": 25,
    "spo2_idx": 0, "bpm_idx": 1, "ir_start_idx": 2, "ir_end_idx": 27,
    "motion_idx": 27, "total_channels": 28,
}
MODEL_CFG = {
    "conv_filters": [16, 32], "kernel_sizes": [5, 3], "strides": [1, 1],
    "pool_after_layer": [True, True], "dense_units": [32], "dropout": 0.3,
}

print("Building placeholder architecture (random weights, no training)...")
model = mu.build_model(WINDOW_FRAMES, N_CHANNELS, MODEL_CFG)

print("Fitting normalization stats on random noise (placeholder only)...")
X_fake = np.random.randn(200, WINDOW_FRAMES, 28).astype(np.float32)
Xc = mu.build_compressed_tensor(X_fake, FEATURES, CH_CFG)
mean = Xc.reshape(-1, N_CHANNELS).mean(axis=0)
std = Xc.reshape(-1, N_CHANNELS).std(axis=0)
std[std == 0] = 1.0
Xn = (Xc - mean) / std

print("Quantizing to int8 .tflite...")
tflite_bytes, in_q, out_q = mu.quantize_int8(model, Xn)

config = {
    "data": {"test_fraction": 0.2},
    "channels": CH_CFG,
    "label": {
        "num_classes": 3, "class_names": ["Normal", "Abnormal", "Emergency"], "class_values": [0, 1, 2],
        "training_target": {"class_targets": [0.0, 0.7, 1.0]},
    },
    "threshold_tuning": {"tuning_fraction": 0.2, "seed": 0, "metric": "f1_macro"},
    "compression": {"name": "dev_placeholder", "features": FEATURES},
    "model_variant": MODEL_CFG,
    "training": {"epochs": 1, "batch_size": 32},
    "normalization": {"mean": mean.tolist(), "std": std.tolist(),
                       "channel_order": FEATURES + ["spo2", "bpm", "motion_level"]},
    "input_quantization": in_q,
    "output_quantization": out_q,
    "thresholds": {"tau_1": 0.33, "tau_2": 0.66, "tuned_for_metric": "f1_macro"},
    "_dev_placeholder": True,
}

with open(OUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)
model.save(str(OUT_DIR / "model_float.h5"))
with open(OUT_DIR / "model_int8.tflite", "wb") as f:
    f.write(tflite_bytes)

print(f"\nWrote placeholder model to {OUT_DIR}/")
print("Now run: curl -X POST http://127.0.0.1:8000/api/models/register-initial")
print("(or click 'Register models/current/ as active model' in the Streamlit sidebar)")
