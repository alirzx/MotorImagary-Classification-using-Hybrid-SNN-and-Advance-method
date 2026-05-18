import os
import time
import hashlib
from typing import List, Dict, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_definitions import (
    preprocess_input,
    load_all_subject_models,
)

# ============================================================
# MOABB / BRAINDCODE IMPORTS
# ============================================================

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    exponential_moving_standardize,
    create_windows_from_events,
)

# ============================================================
# PROJECT ROOT DETECTION
# ============================================================

def get_project_root() -> str:
    """
    Dynamically locate project root containing Results/.
    Makes project portable after clone.
    """

    current = os.path.dirname(os.path.abspath(__file__))

    while True:

        if os.path.exists(os.path.join(current, "Results")):
            return current

        parent = os.path.dirname(current)

        if parent == current:
            break

        current = parent

    raise FileNotFoundError(
        "Could not locate project root containing Results/"
    )


BASE_DIR = get_project_root()

# ============================================================
# PATH SETUP
# ============================================================

MNE_DATA_DIR = os.path.join(BASE_DIR, "mne_data")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

os.environ["MNE_DATA"] = MNE_DATA_DIR

print("=" * 60)
print("[INFO] PROJECT INITIALIZATION")
print("=" * 60)
print(f"[INFO] BASE_DIR      : {BASE_DIR}")
print(f"[INFO] RESULTS_DIR   : {RESULTS_DIR}")
print(f"[INFO] MNE_DATA_DIR  : {MNE_DATA_DIR}")

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"[INFO] DEVICE        : {device}")

if torch.cuda.is_available():
    print(f"[INFO] GPU           : {torch.cuda.get_device_name(0)}")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="BCI Motor Imagery Classification API",
    version="2.1",
)

# ============================================================
# CONSTANTS
# ============================================================

EEG_CHANNELS = 22
EEG_SAMPLES = 1000
VALID_SUBJECTS = list(range(1, 10))

# ============================================================
# Pydantic SCHEMAS
# ============================================================

class EEGInput(BaseModel):
    data: List[List[float]]


class EEGBatch(BaseModel):
    data: List[List[List[float]]]

# ============================================================
# MODEL DISCOVERY
# ============================================================

def discover_subject_models(
    results_dir: str
) -> Dict[int, str]:
    """
    Discover model checkpoints from:
        Results/subject_X/best_model.pth
    """

    if not os.path.isdir(results_dir):

        raise FileNotFoundError(
            f"Results directory not found: {results_dir}"
        )

    subject_paths = {}

    for folder in sorted(os.listdir(results_dir)):

        if not folder.startswith("subject_"):
            continue

        try:
            subject_id = int(folder.split("_")[1])

        except Exception:
            continue

        model_path = os.path.join(
            results_dir,
            folder,
            "best_model.pth"
        )

        if os.path.isfile(model_path):

            subject_paths[subject_id] = model_path

    if len(subject_paths) == 0:

        raise FileNotFoundError(
            "No valid models discovered inside Results/"
        )

    return subject_paths

# ============================================================
# MODEL LOADING
# ============================================================

models: Dict[int, torch.nn.Module] = {}

try:

    subject_model_paths = discover_subject_models(
        RESULTS_DIR
    )

    print("\n[INFO] DISCOVERED MODELS")

    for sid, path in subject_model_paths.items():

        print(f"   Subject {sid} -> {path}")

    models = load_all_subject_models(
        subject_model_paths
    )

    for sid, model in models.items():

        model.to(device)
        model.eval()

        total_params = sum(
            p.numel() for p in model.parameters()
        )

        print(
            f"[INFO] Subject {sid} loaded "
            f"({total_params:,} params)"
        )

    print(
        f"\n[INFO] SUCCESSFULLY LOADED SUBJECTS: "
        f"{sorted(models.keys())}"
    )

except Exception as e:

    print(f"\n[FATAL] MODEL LOADING FAILED")
    print(f"[ERROR] {str(e)}")

    models = {}

# ============================================================
# EEG VALIDATION
# ============================================================

def validate_subject_id(subject_id: int):

    if subject_id not in VALID_SUBJECTS:

        raise HTTPException(
            status_code=400,
            detail=f"Subject ID must be in {VALID_SUBJECTS}"
        )


def validate_eeg_input(
    eeg_data: List[List[float]]
) -> torch.Tensor:

    try:

        x = np.asarray(
            eeg_data,
            dtype=np.float32
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid EEG numeric format"
        )

    if x.shape != (EEG_CHANNELS, EEG_SAMPLES):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected shape "
                f"({EEG_CHANNELS},{EEG_SAMPLES}), "
                f"got {x.shape}"
            )
        )

    if np.isnan(x).any():

        raise HTTPException(
            status_code=400,
            detail="NaN values detected in EEG input"
        )

    if np.isinf(x).any():

        raise HTTPException(
            status_code=400,
            detail="Infinite values detected in EEG input"
        )

    return preprocess_input(x).to(device)

# ============================================================
# BATCH PREPROCESS
# ============================================================

def preprocess_batch(
    windows_np: np.ndarray
) -> torch.Tensor:

    tensors = []

    for window in windows_np:

        t = preprocess_input(
            window.astype(np.float32)
        )

        if t.dim() == 2:
            t = t.unsqueeze(0)

        tensors.append(t.to(device))

    return torch.cat(tensors, dim=0)

# ============================================================
# PREDICTION CACHE
# ============================================================

PRED_CACHE: Dict[str, Dict] = {}

def cache_key(
    window_np: np.ndarray
) -> str:

    h = hashlib.sha1()

    h.update(
        window_np.astype(np.float32).tobytes()
    )

    return h.hexdigest()

# ============================================================
# DATASET CACHE
# ============================================================

DATA_CACHE: Dict[int, Dict] = {}

# ============================================================
# MOABB DATA LOADER
# ============================================================

def load_subject_windows(
    subject_id: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load subject windows using the same preprocessing
    pipeline as training.
    """

    validate_subject_id(subject_id)

    if subject_id in DATA_CACHE:

        print(
            f"[CACHE] Subject {subject_id} dataset cache hit"
        )

        return (
            DATA_CACHE[subject_id]["X"],
            DATA_CACHE[subject_id]["Y"],
        )

    print(f"\n[INFO] Loading MOABB subject {subject_id}")

    # ========================================================
    # LOAD DATASET
    # ========================================================

    try:

        dataset = MOABBDataset(
            dataset_name="BNCI2014_001",
            subject_ids=[subject_id],
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"MOABB load failed: {e}"
        )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocessors = [

        Preprocessor(
            "pick_types",
            eeg=True,
            meg=False,
            stim=False,
        ),

        # convert V -> uV
        Preprocessor(
            lambda data: data * 1e6
        ),

        # mu-beta band
        Preprocessor(
            "filter",
            l_freq=8.0,
            h_freq=30.0,
        ),

        # online-like normalization
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),
    ]

    try:

        preprocess(
            dataset,
            preprocessors,
            n_jobs=1,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {e}"
        )

    # ========================================================
    # WINDOW CREATION
    # ========================================================

    try:

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            preload=True,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Window creation failed: {e}"
        )

    # ========================================================
    # SESSION SPLIT
    # ========================================================

    try:

        split_dict = windows_dataset.split("session")

        if "1test" not in split_dict:

            raise RuntimeError(
                "Session '1test' not found"
            )

        test_set = split_dict["1test"]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Session split failed: {e}"
        )

    # ========================================================
    # EXTRACT WINDOWS
    # ========================================================

    X = []
    Y = []

    for window in test_set:

        x, y = window[0], window[1]

        if isinstance(y, list):
            y = y[0]

        x = np.array(
            x,
            dtype=np.float32
        )

        if x.shape != (EEG_CHANNELS, EEG_SAMPLES):

            print(
                f"[WARNING] Skipping invalid shape: {x.shape}"
            )

            continue

        X.append(x)
        Y.append(int(y))

    DATA_CACHE[subject_id] = {
        "X": X,
        "Y": Y,
    }

    print(
        f"[INFO] Subject {subject_id} loaded "
        f"with {len(X)} windows"
    )

    return X, Y

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "BCI Motor Imagery API Running",
        "device": str(device),
        "loaded_subjects": sorted(models.keys()),
        "model_count": len(models),
        "cache_size": len(PRED_CACHE),
    }

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": len(models) > 0,
    }

# ============================================================
# FETCH SUBJECT
# ============================================================

@app.get("/fetch_subject/{subject_id}")
def fetch_subject(subject_id: int):

    validate_subject_id(subject_id)

    X, Y = load_subject_windows(subject_id)

    return {
        "subject": subject_id,
        "count": len(X),
        "labels": Y,
        "data": [x.tolist() for x in X],
    }

# ============================================================
# FETCH ALL
# ============================================================

@app.get("/fetch_all")
def fetch_all():

    result = {}

    for sid in sorted(models.keys()):

        X, Y = load_subject_windows(sid)

        result[sid] = {
            "subject": sid,
            "count": len(X),
            "labels": Y,
            "data": [x.tolist() for x in X],
        }

    return result

# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict/{subject_id}")
def predict_subject(
    subject_id: int,
    eeg: EEGInput
):

    if subject_id not in models:

        raise HTTPException(
            status_code=404,
            detail=f"Model for subject {subject_id} not loaded"
        )

    x_tensor = validate_eeg_input(eeg.data)

    model = models[subject_id]

    with torch.no_grad():

        logits = model(x_tensor)

        probs = torch.softmax(
            logits,
            dim=-1
        )

        pred = int(
            probs.argmax(dim=-1).item()
        )

    return {
        "subject": subject_id,
        "prediction": pred,
        "probabilities": probs.squeeze(0).cpu().tolist(),
    }

# ============================================================
# PREDICT ALL SUBJECTS
# ============================================================

@app.post("/predict_all")
def predict_all_subjects(
    eeg: EEGInput
):

    x_tensor = validate_eeg_input(eeg.data)

    results = {}

    for sid, model in models.items():

        with torch.no_grad():

            logits = model(x_tensor)

            probs = torch.softmax(
                logits,
                dim=-1
            )

            pred = int(
                probs.argmax(dim=-1).item()
            )

        results[sid] = {
            "subject": sid,
            "prediction": pred,
            "probabilities": probs.squeeze(0).cpu().tolist(),
        }

    return results

# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/predict_batch/{subject_id}")
def predict_batch(
    subject_id: int,
    payload: EEGBatch
):

    if subject_id not in models:

        raise HTTPException(
            status_code=404,
            detail=f"Model for subject {subject_id} not loaded"
        )

    try:

        arr = np.asarray(
            payload.data,
            dtype=np.float32
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid batch numeric format"
        )

    if arr.ndim != 3:

        raise HTTPException(
            status_code=400,
            detail=f"Expected 3D tensor, got {arr.ndim}D"
        )

    if arr.shape[1:] != (EEG_CHANNELS, EEG_SAMPLES):

        raise HTTPException(
            status_code=400,
            detail=f"Invalid shape: {arr.shape}"
        )

    if np.isnan(arr).any():

        raise HTTPException(
            status_code=400,
            detail="NaN values detected"
        )

    model = models[subject_id]

    preds = [None] * len(arr)
    probs = [None] * len(arr)

    cache_hits = 0
    to_compute = []

    start_time = time.time()

    # ========================================================
    # CACHE CHECK
    # ========================================================

    for i in range(len(arr)):

        k = cache_key(arr[i])

        if k in PRED_CACHE:

            preds[i] = PRED_CACHE[k]["pred"]
            probs[i] = PRED_CACHE[k]["probs"]

            cache_hits += 1

        else:

            to_compute.append(i)

    # ========================================================
    # COMPUTE REMAINING
    # ========================================================

    if len(to_compute) > 0:

        batch_tensor = preprocess_batch(
            arr[to_compute]
        )

        with torch.no_grad():

            logits = model(batch_tensor)

            batch_probs = torch.softmax(
                logits,
                dim=-1
            ).cpu().numpy()

        for j, idx in enumerate(to_compute):

            pred = int(
                batch_probs[j].argmax()
            )

            prob = batch_probs[j].tolist()

            preds[idx] = pred
            probs[idx] = prob

            PRED_CACHE[
                cache_key(arr[idx])
            ] = {
                "pred": pred,
                "probs": prob,
            }

    total_time = time.time() - start_time

    return {
        "subject": subject_id,
        "count": int(arr.shape[0]),
        "predictions": preds,
        "probabilities": probs,
        "cache_hits": cache_hits,
        "cache_size": len(PRED_CACHE),
        "batch_infer_time_s": round(total_time, 5),
        "per_window_avg_ms": round(
            (total_time / arr.shape[0]) * 1000.0,
            4
        ),
    }

# ============================================================
# DUMMY INPUT
# ============================================================

@app.get("/dummy")
def dummy():

    dummy_input = np.random.randn(
        EEG_CHANNELS,
        EEG_SAMPLES
    ).astype(np.float32)

    return {
        "shape": list(dummy_input.shape),
        "dummy_input": dummy_input.tolist(),
    }

# ============================================================
# DUMMY SINGLE PREDICTION
# ============================================================

@app.get("/dummy_predict/{subject_id}")
def dummy_predict(subject_id: int):

    if subject_id not in models:

        raise HTTPException(
            status_code=404,
            detail=f"Model for subject {subject_id} not loaded"
        )

    dummy_input = np.random.randn(
        EEG_CHANNELS,
        EEG_SAMPLES
    ).astype(np.float32)

    x = preprocess_input(dummy_input).to(device)

    with torch.no_grad():

        logits = models[subject_id](x)

        probs = torch.softmax(
            logits,
            dim=-1
        )

        pred = int(
            probs.argmax(dim=-1).item()
        )

    return {
        "subject": subject_id,
        "prediction": pred,
        "probabilities": probs.squeeze(0).cpu().tolist(),
    }

# ============================================================
# DUMMY ALL SUBJECTS
# ============================================================

@app.get("/dummy_predict_all")
def dummy_predict_all():

    results = {}

    for sid, model in models.items():

        dummy_input = np.random.randn(
            EEG_CHANNELS,
            EEG_SAMPLES
        ).astype(np.float32)

        x = preprocess_input(dummy_input).to(device)

        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=-1
            )

            pred = int(
                probs.argmax(dim=-1).item()
            )

        results[sid] = {
            "subject": sid,
            "prediction": pred,
            "probabilities": probs.squeeze(0).cpu().tolist(),
        }

    return results

# ============================================================
# CLEAR CACHE
# ============================================================

@app.post("/clear_cache")
def clear_cache():

    global PRED_CACHE

    old_size = len(PRED_CACHE)

    PRED_CACHE = {}

    return {
        "message": "Prediction cache cleared",
        "old_cache_size": old_size,
    }

# ============================================================
# RUN
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