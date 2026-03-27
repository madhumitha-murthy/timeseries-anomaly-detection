"""
api.py — FastAPI REST API for LSTM Autoencoder Anomaly Detection

Why FastAPI?
  • Automatic request / response validation via Pydantic — malformed inputs are
    rejected before they reach model code.
  • Auto-generated Swagger UI at /docs (OpenAPI spec) for immediate interactive
    testing without any extra tooling.
  • Async-native: endpoints can be non-blocking so a slow batch request does not
    starve health-check probes.
  • Minimal boilerplate compared with Flask while offering better type safety.

Expected latency:
  ~12 ms per window on CPU for a single-channel (F=1) window of length 128.
  Most of the budget is the LSTM forward pass (~8 ms); tensor allocation and
  JSON serialisation account for the rest.
"""

import os
import time
from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import LSTMAutoencoder

# ---------------------------------------------------------------------------
# Configuration (override via environment variables at runtime)
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "../models/lstm_ae_best.pth")
INPUT_DIM = int(os.getenv("INPUT_DIM", "1"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "64"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "2"))
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.05"))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LSTM Autoencoder Anomaly Detection API",
    description="Detects anomalies in multivariate time-series windows using an LSTM Autoencoder trained on NASA SMAP satellite telemetry.",
    version="1.0.0",
)

# Global model reference populated at startup
_model: LSTMAutoencoder = None
_device: torch.device = None


@app.on_event("startup")
def load_model():
    """Load the trained LSTM Autoencoder once when the server starts.

    Using a startup event ensures the model is loaded exactly once, shared
    across all requests, and ready before the first request arrives.
    """
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,  # no dropout at inference
    ).to(_device)

    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=_device)
        _model.load_state_dict(state)
        print(f"[startup] Loaded model weights from {MODEL_PATH}")
    else:
        print(f"[startup] WARNING — model file not found at {MODEL_PATH}. Serving with random weights.")

    _model.eval()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class WindowRequest(BaseModel):
    """A single time-series window to score."""

    window: List[List[float]] = Field(
        ...,
        description=(
            "2-D list of shape [seq_len, num_features]. "
            "For SMAP channel P-1 this is [128, 1]."
        ),
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        description="Anomaly score threshold above which is_anomaly=True.",
    )


class AnomalyResponse(BaseModel):
    """Anomaly detection result for a single window."""

    anomaly_score: float = Field(..., description="Mean-squared reconstruction error.")
    is_anomaly: bool = Field(..., description="True if anomaly_score >= threshold.")
    threshold_used: float = Field(..., description="Threshold applied to produce is_anomaly.")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
def health():
    """Liveness probe — returns 200 OK when the server is running."""
    return {"status": "ok"}


@app.get("/info", tags=["Ops"])
def info():
    """Return model configuration metadata."""
    return {
        "model": "LSTMAutoencoder",
        "dataset": "NASA SMAP",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "default_threshold": DEFAULT_THRESHOLD,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=AnomalyResponse, tags=["Inference"])
def predict(request: WindowRequest) -> AnomalyResponse:
    """Score a single time-series window.

    Accepts a 2-D list ``window`` of shape ``[seq_len, num_features]`` and
    returns the MSE reconstruction error together with a binary anomaly flag.
    """
    t_start = time.perf_counter()

    # --- Validate input ---
    window = request.window
    if not window or not isinstance(window[0], list):
        raise HTTPException(status_code=400, detail="window must be a 2-D list [[f1, f2, ...], ...]")

    seq_len = len(window)
    n_features = len(window[0])

    if n_features != INPUT_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {INPUT_DIM} feature(s) per step, got {n_features}.",
        )

    # --- Build tensor: (1, seq_len, features) ---
    import numpy as np  # local import to keep startup fast
    x = torch.tensor(np.array(window, dtype=np.float32)).unsqueeze(0).to(_device)  # (1, W, F)

    # --- Inference ---
    with torch.no_grad():
        recon, _ = _model(x)
        mse = float(((recon - x) ** 2).mean().item())

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return AnomalyResponse(
        anomaly_score=round(mse, 6),
        is_anomaly=mse >= request.threshold,
        threshold_used=request.threshold,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=List[AnomalyResponse], tags=["Inference"])
def predict_batch(requests: List[WindowRequest]) -> List[AnomalyResponse]:
    """Score a batch of time-series windows.

    Processes each window independently and returns a list of
    :class:`AnomalyResponse` objects in the same order.
    """
    return [predict(req) for req in requests]
