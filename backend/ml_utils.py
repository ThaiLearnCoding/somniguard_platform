"""
ml_utils.py

Mirrors the production pipeline in SomniGuard_experiments_3.ipynb:
  raw (60, 28) window
    -> statistical IR compression (per-frame, config-driven)
    -> per-channel normalize (fixed mean/std from the currently-deployed model)
    -> Conv1D-only stack -> single sigmoid neuron (severity score in [0,1])
    -> tau_1 / tau_2 thresholds -> 3-class label {0: Normal, 1: Abnormal, 2: Emergency}
    -> int8 quantized .tflite for the EFR32xG26 (Conv1D/Pooling/Dense only)

Fine-tuning (Phase 3) loads the PARENT model's float weights (does not reinit / does
not freeze layers), continues training with a conservative LR + EarlyStopping, then
re-tunes tau_1/tau_2 on a held-out split and re-quantizes to int8.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must be set before importing tensorflow

import json
import time
import shutil
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support,
)

CLASS_NAMES = ["Normal", "Abnormal", "Emergency"]
CLASS_VALUES = [0, 1, 2]


class MLError(RuntimeError):
    """Raised for model/config problems (missing files, shape mismatches, etc.)."""


# --------------------------------------------------------------------------------------
# Compression (verbatim logic from the notebook's compress_frame_stats/build_compressed_tensor)
# --------------------------------------------------------------------------------------

def compress_frame_stats(X_ir: np.ndarray, features: List[str]) -> np.ndarray:
    """X_ir: (..., frame_samples). Returns (..., len(features))."""
    mean = X_ir.mean(axis=-1)
    std = X_ir.std(axis=-1)
    std_safe = np.where(std < 1e-8, 1.0, std)
    mn = X_ir.min(axis=-1)
    mx = X_ir.max(axis=-1)

    out = []
    for f in features:
        if f == "mean":
            out.append(mean)
        elif f == "std":
            out.append(std)
        elif f == "min":
            out.append(mn)
        elif f == "max":
            out.append(mx)
        elif f == "range":
            out.append(mx - mn)
        elif f == "rms":
            out.append(np.sqrt(np.mean(X_ir ** 2, axis=-1)))
        elif f == "skew":
            z = (X_ir - mean[..., None]) / std_safe[..., None]
            out.append(np.mean(z ** 3, axis=-1))
        elif f == "kurtosis":
            z = (X_ir - mean[..., None]) / std_safe[..., None]
            out.append(np.mean(z ** 4, axis=-1) - 3.0)
        elif f == "mad":
            out.append(np.mean(np.abs(X_ir - mean[..., None]), axis=-1))
        elif f == "mean_abs_diff":
            out.append(np.mean(np.abs(np.diff(X_ir, axis=-1)), axis=-1))
        else:
            raise MLError(f"Unsupported compression feature '{f}'")
    return np.stack(out, axis=-1).astype(np.float32)


def build_compressed_tensor(X_raw: np.ndarray, features: List[str], ch_cfg: Dict) -> np.ndarray:
    """
    X_raw: (N, window_frames, total_channels) raw windows with channel layout
      [spo2, bpm, ir_0..ir_24, motion_level] (indices from config.json's `channels` block).
    Returns (N, window_frames, len(features) + 3) with channel order
      [*compressed_ir_features, spo2, bpm, motion_level] -- matches the notebook's
      normalization.channel_order convention exactly.
    """
    spo2_idx = ch_cfg["spo2_idx"]
    bpm_idx = ch_cfg["bpm_idx"]
    ir_start = ch_cfg["ir_start_idx"]
    ir_end = ch_cfg["ir_end_idx"]
    motion_idx = ch_cfg["motion_idx"]

    if X_raw.shape[-1] != ch_cfg["total_channels"]:
        raise MLError(
            f"Raw window has {X_raw.shape[-1]} channels but config.json channels.total_channels "
            f"= {ch_cfg['total_channels']}."
        )

    X_ir = X_raw[..., ir_start:ir_end]  # (N, T, 25)
    compressed = compress_frame_stats(X_ir, features)  # (N, T, len(features))

    spo2 = X_raw[..., spo2_idx:spo2_idx + 1]
    bpm = X_raw[..., bpm_idx:bpm_idx + 1]
    motion = X_raw[..., motion_idx:motion_idx + 1]

    return np.concatenate([compressed, spo2, bpm, motion], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Ordinal regression target / threshold tuning (verbatim logic from the notebook)
# --------------------------------------------------------------------------------------

def ordinal_regression_target(y: np.ndarray, class_targets: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    return class_targets[y].astype(np.float32)


def scores_to_labels(scores: np.ndarray, tau_1: float, tau_2: float) -> np.ndarray:
    scores = np.asarray(scores)
    labels = np.zeros_like(scores, dtype=np.int64)
    labels[scores >= tau_1] = 1
    labels[scores >= tau_2] = 2
    return labels


def _score_metric(y_true, y_pred, metric: str) -> float:
    if metric == "f1_macro":
        return f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    raise MLError(f"Unsupported threshold_tuning.metric '{metric}'")


def tune_thresholds(y_true: np.ndarray, scores: np.ndarray, metric: str = "f1_macro",
                     n_candidates: int = 50, seed: int = 0) -> Tuple[float, float, float]:
    """Grid-search tau_1 <= tau_2 over quantiles of the observed score distribution."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    qs = np.linspace(0.01, 0.99, n_candidates)
    candidates = np.unique(np.quantile(scores, qs))
    if len(candidates) < 2:
        candidates = np.array([0.33, 0.66])

    best = (candidates[0], candidates[-1], -1.0)
    for i, t1 in enumerate(candidates):
        for t2 in candidates[i:]:
            preds = scores_to_labels(scores, t1, t2)
            score = _score_metric(y_true, preds, metric)
            if score > best[2]:
                best = (float(t1), float(t2), float(score))
    return best


# --------------------------------------------------------------------------------------
# Model architecture (verbatim from the notebook's build_model)
# --------------------------------------------------------------------------------------

def build_model(window_frames: int, n_channels: int, model_cfg: Dict) -> tf.keras.Model:
    layers = [Input(shape=(window_frames, n_channels))]
    for filters, kernel, stride, do_pool in zip(
        model_cfg["conv_filters"], model_cfg["kernel_sizes"], model_cfg["strides"], model_cfg["pool_after_layer"]
    ):
        layers.append(Conv1D(filters, kernel_size=kernel, strides=stride, padding="same", activation="relu"))
        if do_pool:
            layers.append(MaxPooling1D(2))
    layers.append(GlobalAveragePooling1D())
    for units in model_cfg["dense_units"]:
        layers.append(Dense(units, activation="relu"))
    layers.append(Dropout(model_cfg["dropout"]))
    layers.append(Dense(1, activation="sigmoid"))
    model = Sequential(layers)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# --------------------------------------------------------------------------------------
# Loading a model version (config.json + float .h5 + int8 .tflite) from disk
# --------------------------------------------------------------------------------------

class ModelVersion:
    """
    Wraps one model version's directory. Expected layout (see README for how to
    plug in your real model):
        <version_dir>/
            config.json       # channels/label/compression/model_variant/normalization/
                                # input_quantization/output_quantization/thresholds
            model_float.h5    # trainable Keras checkpoint (fine-tuning starts here)
            model_int8.tflite # deployed, on-device model (matches config's quant params)
    """

    def __init__(self, version_dir: str):
        self.dir = Path(version_dir)
        config_path = self.dir / "config.json"
        if not config_path.exists():
            raise MLError(f"Missing config.json in {self.dir}")
        with open(config_path) as f:
            self.config = json.load(f)

        self.float_path = self.dir / "model_float.h5"
        self.tflite_path = self.dir / "model_int8.tflite"
        if not self.float_path.exists():
            raise MLError(
                f"Missing model_float.h5 in {self.dir}. A trainable float checkpoint is required "
                f"for fine-tuning even if only the int8 .tflite is deployed on-device."
            )

        self._float_model = None
        self._interpreter = None

    @property
    def channels(self) -> Dict:
        return self.config["channels"]

    @property
    def compression_features(self) -> List[str]:
        return self.config["compression"]["features"]

    @property
    def model_variant(self) -> Dict:
        return self.config["model_variant"]

    @property
    def class_targets(self) -> np.ndarray:
        return np.array(self.config["label"]["training_target"]["class_targets"], dtype=np.float32)

    @property
    def mean(self) -> np.ndarray:
        return np.array(self.config["normalization"]["mean"], dtype=np.float32)

    @property
    def std(self) -> np.ndarray:
        return np.array(self.config["normalization"]["std"], dtype=np.float32)

    @property
    def tau_1(self) -> float:
        return float(self.config["thresholds"]["tau_1"])

    @property
    def tau_2(self) -> float:
        return float(self.config["thresholds"]["tau_2"])

    @property
    def float_model(self) -> tf.keras.Model:
        if self._float_model is None:
            self._float_model = load_model(str(self.float_path), compile=False)
        return self._float_model

    def has_tflite(self) -> bool:
        return self.tflite_path.exists()

    def normalize(self, X_compressed: np.ndarray) -> np.ndarray:
        return (X_compressed - self.mean) / self.std

    def predict_scores_float(self, X_raw: np.ndarray) -> np.ndarray:
        """Raw window (N, T, total_channels) -> severity scores in [0,1] via the float model."""
        X_c = build_compressed_tensor(X_raw, self.compression_features, self.channels)
        X_n = self.normalize(X_c)
        return self.float_model.predict(X_n, verbose=0).ravel()

    def predict_scores_tflite(self, X_raw: np.ndarray) -> np.ndarray:
        """Same as predict_scores_float but runs the deployed int8 .tflite (what's actually on-device)."""
        if not self.has_tflite():
            raise MLError(f"No model_int8.tflite in {self.dir}; falling back requires predict_scores_float.")
        if self._interpreter is None:
            self._interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
            self._interpreter.allocate_tensors()
        interp = self._interpreter
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]
        in_scale, in_zp = in_detail["quantization"]
        out_scale, out_zp = out_detail["quantization"]

        X_c = build_compressed_tensor(X_raw, self.compression_features, self.channels)
        X_n = self.normalize(X_c)

        scores = np.zeros(X_n.shape[0], dtype=np.float32)
        for i in range(X_n.shape[0]):
            sample = X_n[i:i + 1]
            if in_scale:
                q = np.round(sample / in_scale + in_zp).astype(in_detail["dtype"])
            else:
                q = sample.astype(in_detail["dtype"])
            interp.set_tensor(in_detail["index"], q)
            interp.invoke()
            out = interp.get_tensor(out_detail["index"])
            if out_scale:
                out = (out.astype(np.float32) - out_zp) * out_scale
            scores[i] = out.ravel()[0]
        return scores

    def predict_labels(self, X_raw: np.ndarray, prefer_tflite: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (scores, labels) using tflite if available (matches on-device behavior), else float."""
        if prefer_tflite and self.has_tflite():
            scores = self.predict_scores_tflite(X_raw)
        else:
            scores = self.predict_scores_float(X_raw)
        labels = scores_to_labels(scores, self.tau_1, self.tau_2)
        return scores, labels


# --------------------------------------------------------------------------------------
# Windowing: turn per-timestep feature steps + effective labels into (N, 60, 28) + (N,)
# --------------------------------------------------------------------------------------

def predict_over_segments(
    model_version: "ModelVersion",
    features_by_segment: Dict,
    window_frames: int,
    prefer_tflite: bool = True,
) -> Dict:
    """
    Runs inference over every segment's steps, causally: each 60-step window's
    prediction is assigned to the LAST step it covers (since a window is a 30s
    trailing lookback ending "now"). The first `window_frames - 1` steps of each
    segment have no prediction yet (not enough causal history) and map to None.

    features_by_segment: {segment_key: [feature_vector_28, ...]} ordered by step index.
      segment_key can be any hashable (e.g. an int, or a (session_id, segment_index) tuple).
    Returns {segment_key: [(score, label) or None, ...]} same length/order as input.
    """
    out = {}
    for seg_key, feats in features_by_segment.items():
        n = len(feats)
        result = [None] * n
        if n >= window_frames:
            arr = np.array(feats, dtype=np.float32)
            windows = np.stack([arr[i:i + window_frames] for i in range(n - window_frames + 1)])
            scores, labels = model_version.predict_labels(windows, prefer_tflite=prefer_tflite)
            for i in range(len(scores)):
                end_step = i + window_frames - 1
                result[end_step] = (float(scores[i]), int(labels[i]))
        out[seg_key] = result
    return out


def build_windows_from_steps(
    steps_by_segment: Dict[int, List[Tuple[List[float], Optional[int]]]],
    window_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    steps_by_segment: {segment_index: [(feature_vector_28, effective_label_or_None), ...]}
      ordered by step_index within each segment.
    Returns:
      X: (N, window_frames, 28) float32
      y: (N,) int64 -- majority-vote label over the window's covered steps (ties -> more severe)
      end_step_global_index: (N,) int -- convenience index of the last step covered (for
        mapping window predictions back onto the timeline / for testing)
    """
    X_list, y_list, end_idx_list = [], [], []
    global_offset = 0
    for seg_idx in sorted(steps_by_segment.keys()):
        rows = steps_by_segment[seg_idx]
        n = len(rows)
        if n < window_frames:
            global_offset += n
            continue
        feats = np.array([r[0] for r in rows], dtype=np.float32)
        labels = [r[1] for r in rows]

        for start in range(0, n - window_frames + 1):
            end = start + window_frames
            window_labels = [l for l in labels[start:end] if l is not None]
            if not window_labels:
                continue  # no labels at all for this window yet (e.g. brand new upload, no inference run)
            counts = {c: window_labels.count(c) for c in set(window_labels)}
            max_count = max(counts.values())
            tied = sorted([c for c, cnt in counts.items() if cnt == max_count], reverse=True)
            majority_label = tied[0]  # tie-break toward the MORE SEVERE class, deliberately (safety-first)

            X_list.append(feats[start:end])
            y_list.append(majority_label)
            end_idx_list.append(global_offset + end - 1)

        global_offset += n

    if not X_list:
        raise MLError("No complete windows could be built (not enough labeled, contiguous steps).")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
        np.array(end_idx_list, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# Fine-tuning (Phase 3)
# --------------------------------------------------------------------------------------

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": CLASS_NAMES,
        "n_windows": int(len(y_true)),
    }


def quantize_int8(float_model: tf.keras.Model, representative_X: np.ndarray) -> Tuple[bytes, Dict, Dict]:
    """Standard full-integer post-training quantization, representative data drawn from the fit split."""

    def rep_dataset():
        n = min(len(representative_X), 300)
        idx = np.random.RandomState(0).choice(len(representative_X), size=n, replace=False)
        for i in idx:
            yield [representative_X[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(float_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    in_scale, in_zp = in_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]

    return (
        tflite_bytes,
        {"scale": float(in_scale), "zero_point": int(in_zp)},
        {"scale": float(out_scale), "zero_point": int(out_zp)},
    )


def finetune(
    parent: ModelVersion,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    output_dir: str,
    learning_rate: float = 1e-5,
    epochs: int = 200,
    batch_size: int = 32,
    early_stopping_patience: int = 15,
    test_fraction: Optional[float] = None,
    tuning_fraction: Optional[float] = None,
    seed: int = 42,
) -> Dict:
    """
    Fine-tunes the PARENT model's float weights (no layer freezing) on newly
    corrected data, using a conservative LR + EarlyStopping on a held-out split,
    then re-tunes tau_1/tau_2 and re-quantizes to int8.

    Reuses the parent's normalization (mean/std) so the existing float weights'
    learned scale stays meaningful -- only the weights adapt to this patient.

    Returns a dict with the new ModelVersion's file paths + full comparison metrics
    (new candidate vs. the parent, evaluated on the SAME held-out test split).
    """
    if len(np.unique(y_raw)) < 2:
        raise MLError(
            f"Need at least 2 distinct classes to fine-tune / tune thresholds; got only "
            f"{sorted(set(y_raw.tolist()))}. Label more of the timeline before training."
        )

    cfg = parent.config
    test_fraction = test_fraction if test_fraction is not None else cfg.get("data", {}).get("test_fraction", 0.2)
    tuning_fraction = tuning_fraction if tuning_fraction is not None else cfg.get("threshold_tuning", {}).get("tuning_fraction", 0.2)

    # Guard against tiny personalization datasets where a stratified 3-way split isn't possible.
    min_per_class = min(np.bincount(y_raw))
    can_stratify = min_per_class >= 2

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_raw, y_raw, test_size=test_fraction, random_state=seed,
        stratify=y_raw if can_stratify else None,
    )
    can_stratify_2 = can_stratify and min(np.bincount(y_trainval)) >= 2
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_trainval, y_trainval, test_size=tuning_fraction, random_state=seed,
        stratify=y_trainval if can_stratify_2 else None,
    )

    features = parent.compression_features
    ch_cfg = parent.channels
    mean, std = parent.mean, parent.std

    def prep(X):
        Xc = build_compressed_tensor(X, features, ch_cfg)
        return (Xc - mean) / std

    X_train_n, X_tune_n, X_test_n = prep(X_train), prep(X_tune), prep(X_test)

    class_targets = parent.class_targets
    y_train_target = ordinal_regression_target(y_train, class_targets)
    y_tune_target = ordinal_regression_target(y_tune, class_targets)

    # Load parent's float weights and continue training (NOT a fresh build_model() init).
    model = tf.keras.models.clone_model(parent.float_model)
    model.set_weights(parent.float_model.get_weights())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )
    history = model.fit(
        X_train_n, y_train_target,
        validation_data=(X_tune_n, y_tune_target),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=0,
    )

    tune_scores = model.predict(X_tune_n, verbose=0).ravel()
    metric = cfg.get("threshold_tuning", {}).get("metric", "f1_macro")
    tau_1, tau_2, tuning_score = tune_thresholds(y_tune, tune_scores, metric=metric)

    test_scores = model.predict(X_test_n, verbose=0).ravel()
    test_pred = scores_to_labels(test_scores, tau_1, tau_2)
    new_metrics = evaluate_predictions(y_test, test_pred)
    new_metrics["tau_1"] = tau_1
    new_metrics["tau_2"] = tau_2
    new_metrics["tuning_score"] = tuning_score
    new_metrics["epochs_run"] = len(history.history.get("loss", []))
    new_metrics["n_train_windows"] = int(len(y_train))
    new_metrics["n_tune_windows"] = int(len(y_tune))

    # Fair comparison: evaluate the PARENT model on this same fresh test split.
    parent_test_scores = parent.float_model.predict(X_test_n, verbose=0).ravel()
    parent_test_pred = scores_to_labels(parent_test_scores, parent.tau_1, parent.tau_2)
    parent_metrics = evaluate_predictions(y_test, parent_test_pred)
    parent_metrics["tau_1"] = parent.tau_1
    parent_metrics["tau_2"] = parent.tau_2

    tflite_bytes, in_quant, out_quant = quantize_int8(model, X_train_n)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    float_path = out_dir / "model_float.h5"
    tflite_path = out_dir / "model_int8.tflite"
    model.save(str(float_path))
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    new_config = json.loads(json.dumps(cfg))  # deep copy
    new_config["thresholds"] = {
        "tau_1": tau_1, "tau_2": tau_2,
        "tuned_for_metric": metric,
        "score_domain": "float sigmoid output in [0, 1], AFTER dequantizing the int8 model output",
        "decision_rule": "score < tau_1 -> Normal (0); tau_1 <= score < tau_2 -> Abnormal (1); score >= tau_2 -> Emergency (2)",
    }
    new_config["input_quantization"] = in_quant
    new_config["output_quantization"] = out_quant
    new_config["fine_tuning"] = {
        "parent_dir": str(parent.dir),
        "learning_rate": learning_rate,
        "epochs_run": new_metrics["epochs_run"],
        "batch_size": batch_size,
        "trained_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    return {
        "version_dir": str(out_dir),
        "float_path": str(float_path),
        "tflite_path": str(tflite_path),
        "config_path": str(out_dir / "config.json"),
        "new_metrics": new_metrics,
        "parent_metrics": parent_metrics,
        "n_test_windows": int(len(y_test)),
    }
