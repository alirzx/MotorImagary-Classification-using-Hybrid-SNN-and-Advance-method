import os
# Set MNE_DATA before anything else
os.environ['MNE_DATA'] = r'D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\MI_API'

import time
import hashlib
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from model_definitions import SpiTranNet, preprocess_input, load_all_subject_models

# ----------------------------------------
# NEW IMPORTS FOR REAL DATASET ENDPOINT
# ----------------------------------------
from moabb.datasets import BNCI2014_001
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    exponential_moving_standardize,
    create_windows_from_events,
)
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize
from braindecode.preprocessing import SetEEGReference
import pandas as pd
# ----------------------------------------

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Subject model paths
# ----------------------------
subject_model_paths = {
    1: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_1\best_model.pth",
    2: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_2\best_model.pth",
    3: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_3\best_model.pth",
    4: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_4\best_model.pth",
    5: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_5\best_model.pth",
    6: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_6\best_model.pth",
    7: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_7\best_model.pth",
    8: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_8\best_model.pth",
    9: r"D:\my_projects\MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method\our_origin_paper\our_final_results\our_final_results\subject_9\best_model.pth",
}

# ----------------------------
# Load all models
# ----------------------------
try:
    models = load_all_subject_models(subject_model_paths)
    # ensure models on device and eval mode
    for sid, m in models.items():
        m.to(device)
        m.eval()
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    models = {}

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="BCI Motor Imagery Classification API", version="1.0")

# ----------------------------
# Pydantic models
# ----------------------------
class EEGInput(BaseModel):
    data: List[List[float]]  # one window: [22,1000]

class EEGBatch(BaseModel):
    data: List[List[List[float]]]  # batch: [[22,1000], [22,1000], ...]

# ----------------------------
# Helper: validate single EEG input
# ----------------------------
def validate_eeg_input(eeg_data: List[List[float]]) -> torch.Tensor:
    x = np.array(eeg_data, dtype=np.float32)
    if x.shape != (22, 1000):
        raise HTTPException(status_code=400, detail="Input must have shape [22, 1000]")
    return preprocess_input(x).to(device)  # ensure on device

# ----------------------------
# Batch preprocessing wrapper
# ----------------------------
def preprocess_input_batch(windows_np: np.ndarray) -> torch.Tensor:
    """
    windows_np: (N, 22, 1000) numpy float32
    Uses preprocess_input for each window and stacks results.
    preprocess_input is expected to return shape (1, C, L) or (B, C, L).
    """
    tensors = []
    for w in windows_np:
        t = preprocess_input(w.astype(np.float32))
        # ensure batch dimension exists
        if t.dim() == 2:
            t = t.unsqueeze(0)
        tensors.append(t.to(device))
    return torch.cat(tensors, dim=0)

# ----------------------------
# Simple in-memory cache to avoid recomputing identical windows
# key: sha1(window.tobytes()), value: (pred_class, probs_list)
# ----------------------------
PRED_CACHE: Dict[str, Dict] = {}

def cache_key_for_window(window_np: np.ndarray) -> str:
    h = hashlib.sha1()
    # normalize bytes for deterministic hashing
    h.update(window_np.astype(np.float32).tobytes())
    return h.hexdigest()

# ============================================================
# Real data loader (same as before - using MOABBDataset)
# ============================================================
def load_session2_windows(subject_id: int):
    """Load TEST windows exactly like training notebook."""
    try:
        dataset = MOABBDataset(dataset_name="BNCI2014_001",
                               subject_ids=[subject_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ----------------------------
    # Preprocessing (same as training)
    # ----------------------------
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: data * 1e6),
        Preprocessor('filter', l_freq=8., h_freq=30.),
        Preprocessor(exponential_moving_standardize,
                     factor_new=1e-3, init_block_size=1000),
        SetEEGReference()
    ]

    try:
        preprocess(dataset, preprocessors, n_jobs=-1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocess failed: {e}")

    # ----------------------------
    # Create windows like training
    # ----------------------------
    try:
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            preload=True
        )
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Window creation failed: {e}")

    # ----------------------------
    # Split by session → EXACTLY like training
    # ----------------------------
    try:
        test_set = windows_dataset.split('session')['1test']
    except KeyError:
        raise HTTPException(status_code=500,
                            detail=f"Subject {subject_id}: missing session '1test'")

    # ----------------------------
    # Extract samples
    # ----------------------------
    X, Y = [], []

    for window in test_set:
        x, y = window[0], window[1]
        if isinstance(y, list):
            y = y[0]

        if int(y) in [0, 1]:
            X.append(np.array(x, dtype=np.float32))
            Y.append(int(y))

    return X, Y

# -----------------------------------------------------------------
# Existing single-window prediction endpoint (unchanged)
# but keep for backward compatibility
# -----------------------------------------------------------------
@app.post("/predict/{subject_id}")
def predict_subject(subject_id: int, eeg: EEGInput):
    if subject_id not in models:
        raise HTTPException(status_code=404, detail=f"Subject {subject_id} model not loaded")
    x_tensor = validate_eeg_input(eeg.data)
    with torch.no_grad():
        logits = models[subject_id](x_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(dim=-1).item())
    return {
        "subject": subject_id,
        "prediction": pred_class,
        "probabilities": probs.squeeze(0).tolist()
    }

# -----------------------------------------------------------------
# NEW: batch predict endpoint for a single subject (fast)
# Accepts many windows at once and uses batching and cache.
# -----------------------------------------------------------------
@app.post("/predict_batch/{subject_id}")
def predict_batch_subject(subject_id: int, payload: EEGBatch):
    if subject_id not in models:
        raise HTTPException(status_code=404, detail=f"Subject {subject_id} model not loaded")

    data_list = payload.data
    if not isinstance(data_list, list) or len(data_list) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")

    # Validate shapes
    try:
        arr = np.stack([np.asarray(w, dtype=np.float32) for w in data_list], axis=0)  # (N,22,1000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid batch data: {e}")

    if arr.ndim != 3 or arr.shape[1:] != (22, 1000):
        raise HTTPException(status_code=400, detail=f"Each window must have shape [22,1000], got {arr.shape}")

    model = models[subject_id]

    preds = []
    probs = []
    cache_hits = 0
    start_time = time.time()

    # first check cache per-window and build list to compute
    to_compute_idx = []
    for i in range(arr.shape[0]):
        key = cache_key_for_window(arr[i])
        if key in PRED_CACHE:
            cache_entry = PRED_CACHE[key]
            preds.append(cache_entry["pred"])
            probs.append(cache_entry["probs"])
            cache_hits += 1
        else:
            preds.append(None)
            probs.append(None)
            to_compute_idx.append(i)

    # compute for those not in cache in one batch
    if len(to_compute_idx) > 0:
        sub_windows = arr[to_compute_idx]  # (M,22,1000)
        batch_tensor = preprocess_input_batch(sub_windows)  # shape (M, C, L)
        with torch.no_grad():
            logits = model(batch_tensor)
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (M, num_classes)
            batch_preds = batch_probs.argmax(axis=-1).tolist()
        # fill results and cache
        for idx_pos, idx in enumerate(to_compute_idx):
            p = int(batch_preds[idx_pos])
            pr = batch_probs[idx_pos].tolist()
            key = cache_key_for_window(arr[idx])
            PRED_CACHE[key] = {"pred": p, "probs": pr}
            preds[idx] = p
            probs[idx] = pr

    total_time = time.time() - start_time
    per_window_avg_ms = (total_time / float(arr.shape[0])) * 1000.0

    return {
        "subject": subject_id,
        "count": int(arr.shape[0]),
        "predictions": preds,
        "probabilities": probs,
        "cache_hits": int(cache_hits),
        "batch_infer_time_s": total_time,
        "per_window_avg_ms": per_window_avg_ms
    }

# -----------------------------------------------------------------
# NEW: batch predict for all loaded subjects (one HTTP call)
# -----------------------------------------------------------------
@app.post("/predict_batch_all")
def predict_batch_all(payload: EEGBatch):
    """
    payload.data is expected to be a list with one entry per subject:
    [{"subject": sid, "data": [[22,1000], ...]}, ...]
    For backward compatibility we also accept a dict keyed by subject as JSON,
    but main usage from Streamlit will call per-subject endpoint instead.
    """
    raise HTTPException(status_code=501, detail="Use per-subject /predict_batch/{subject_id} from frontend for now.")

# -----------------------------------------------------------------
# Existing endpoints to fetch real data (unchanged)
# -----------------------------------------------------------------
@app.get("/fetch_subject/{subject_id}")
def fetch_subject(subject_id: int):
    return get_real_data(str(subject_id))

@app.get("/fetch_all")
def fetch_all():
    return get_real_data("all")

@app.get("/get_real_data/{subject_id}")
def get_real_data(subject_id: str):
    """Fetch real BCI session-2 windows for one subject OR all."""
    if subject_id == "all":
        all_data = {}
        for sid in range(1, 10):
            X, Y = load_session2_windows(sid)
            all_data[sid] = {
                "data": [x.tolist() for x in X],  # send serializable numpy -> list
                "labels": Y,
                "count": len(X)
            }
        return all_data
    else:
        sid = int(subject_id)
        if sid < 1 or sid > 9:
            raise HTTPException(status_code=400, detail="Subject ID must be 1–9")
        X, Y = load_session2_windows(sid)
        return {
            "subject": sid,
            "data": [x.tolist() for x in X],
            "labels": Y,
            "count": len(X)
        }

# -----------------------------------------------------------------
# Dummy endpoints (unchanged)
# -----------------------------------------------------------------
@app.get("/dummy")
def dummy_prediction():
    dummy_input = np.random.randn(22, 1000).tolist()
    return {"dummy_input": dummy_input}
@app.get("/dummy_predict/{subject_id}")
def dummy_predict_subject(subject_id: int):
    if subject_id not in models:
        raise HTTPException(status_code=404, detail=f"Subject {subject_id} model not loaded")
    
    dummy_input = np.random.randn(22, 1000)
    x_tensor = preprocess_input(dummy_input)
    with torch.no_grad():
        logits = models[subject_id](x_tensor.to(device))
        probs = torch.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(dim=-1).item())

    # Return only prediction and probabilities
    return {
        "subject": subject_id,
        "prediction": pred_class,
        "probabilities": probs.squeeze(0).tolist()
    }

@app.get("/dummy_predict_all")
def dummy_predict_all():
    results = {}
    for subject_id in models.keys():
        dummy_input = np.random.randn(22, 1000)
        x_tensor = preprocess_input(dummy_input)
        with torch.no_grad():
            logits = models[subject_id](x_tensor.to(device))
            probs = torch.softmax(logits, dim=-1)
            pred_class = int(probs.argmax(dim=-1).item())

        results[subject_id] = {
            "subject": subject_id,
            "prediction": pred_class,
            "probabilities": probs.squeeze(0).tolist()
        }
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="debug"
    )
