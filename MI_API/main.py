import os
import time
import hashlib
from typing import List, Dict

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_definitions import (
    SpiTranNet,
    preprocess_input,
    load_all_subject_models,
)

# ----------------------------------------
# PATH SETUP (PORTABLE)
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Store MOABB data locally (portable)
MNE_DATA_DIR = os.path.join(BASE_DIR, "mne_data")
os.environ["MNE_DATA"] = MNE_DATA_DIR

# Results directory (models + artifacts)
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# ----------------------------------------
# Device setup
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# Dynamic model discovery
# ----------------------------------------
def discover_subject_models(results_dir: str) -> Dict[int, str]:
    """
    Automatically discover subject models from:
    Results/subject_X/best_model.pth
    """
    subject_paths = {}

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for name in sorted(os.listdir(results_dir)):
        if not name.startswith("subject_"):
            continue

        try:
            subject_id = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue

        model_path = os.path.join(results_dir, name, "best_model.pth")
        if os.path.isfile(model_path):
            subject_paths[subject_id] = model_path

    if len(subject_paths) == 0:
        raise FileNotFoundError("No subject models found in Results/")

    return subject_paths


# ----------------------------------------
# Load all models (AUTO)
# ----------------------------------------
try:
    subject_model_paths = discover_subject_models(RESULTS_DIR)
    models = load_all_subject_models(subject_model_paths)

    for sid, model in models.items():
        model.to(device)
        model.eval()

    print(f"[INFO] Loaded models for subjects: {sorted(models.keys())}")

except Exception as e:
    print(f"[FATAL] Model loading failed: {e}")
    models = {}

# ----------------------------------------
# FastAPI app
# ----------------------------------------
app = FastAPI(
    title="BCI Motor Imagery Classification API",
    version="1.0",
)

# ----------------------------------------
# Pydantic models
# ----------------------------------------
class EEGInput(BaseModel):
    data: List[List[float]]  # [22,1000]


class EEGBatch(BaseModel):
    data: List[List[List[float]]]  # [[22,1000], ...]


# ----------------------------------------
# Validation helpers
# ----------------------------------------
def validate_eeg_input(eeg_data: List[List[float]]) -> torch.Tensor:
    x = np.asarray(eeg_data, dtype=np.float32)
    if x.shape != (22, 1000):
        raise HTTPException(
            status_code=400,
            detail="EEG input must have shape [22, 1000]",
        )
    return preprocess_input(x).to(device)


def preprocess_input_batch(windows_np: np.ndarray) -> torch.Tensor:
    tensors = []
    for w in windows_np:
        t = preprocess_input(w.astype(np.float32))
        if t.dim() == 2:
            t = t.unsqueeze(0)
        tensors.append(t.to(device))
    return torch.cat(tensors, dim=0)


# ----------------------------------------
# Prediction cache
# ----------------------------------------
PRED_CACHE: Dict[str, Dict] = {}


def cache_key_for_window(window_np: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(window_np.astype(np.float32).tobytes())
    return h.hexdigest()


# ============================================================
# Prediction endpoints
# ============================================================
@app.post("/predict/{subject_id}")
def predict_subject(subject_id: int, eeg: EEGInput):
    if subject_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model for subject {subject_id} not found",
        )

    x_tensor = validate_eeg_input(eeg.data)

    with torch.no_grad():
        logits = models[subject_id](x_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred = int(probs.argmax(dim=-1).item())

    return {
        "subject": subject_id,
        "prediction": pred,
        "probabilities": probs.squeeze(0).tolist(),
    }


@app.post("/predict_batch/{subject_id}")
def predict_batch_subject(subject_id: int, payload: EEGBatch):
    if subject_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model for subject {subject_id} not found",
        )

    arr = np.asarray(payload.data, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (22, 1000):
        raise HTTPException(
            status_code=400,
            detail="Each window must have shape [22, 1000]",
        )

    model = models[subject_id]

    preds, probs = [], []
    cache_hits = 0
    to_compute = []

    start = time.time()

    for i in range(arr.shape[0]):
        key = cache_key_for_window(arr[i])
        if key in PRED_CACHE:
            preds.append(PRED_CACHE[key]["pred"])
            probs.append(PRED_CACHE[key]["probs"])
            cache_hits += 1
        else:
            preds.append(None)
            probs.append(None)
            to_compute.append(i)

    if to_compute:
        batch = preprocess_input_batch(arr[to_compute])
        with torch.no_grad():
            logits = model(batch)
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_preds = batch_probs.argmax(axis=-1)

        for j, idx in enumerate(to_compute):
            preds[idx] = int(batch_preds[j])
            probs[idx] = batch_probs[j].tolist()
            PRED_CACHE[cache_key_for_window(arr[idx])] = {
                "pred": preds[idx],
                "probs": probs[idx],
            }

    total_time = time.time() - start

    return {
        "subject": subject_id,
        "count": int(arr.shape[0]),
        "predictions": preds,
        "probabilities": probs,
        "cache_hits": cache_hits,
        "batch_infer_time_s": total_time,
        "per_window_avg_ms": (total_time / arr.shape[0]) * 1000.0,
    }


# ============================================================
# Dummy endpoints (unchanged behavior)
# ============================================================
@app.get("/dummy_predict/{subject_id}")
def dummy_predict_subject(subject_id: int):
    if subject_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model for subject {subject_id} not found",
        )

    dummy = np.random.randn(22, 1000).astype(np.float32)
    x = preprocess_input(dummy).to(device)

    with torch.no_grad():
        logits = models[subject_id](x)
        probs = torch.softmax(logits, dim=-1)
        pred = int(probs.argmax(dim=-1).item())

    return {
        "subject": subject_id,
        "prediction": pred,
        "probabilities": probs.squeeze(0).tolist(),
    }


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
