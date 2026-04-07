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
  ~12 ms per window on CPU for a 25-feature window of length 30 (E-7 config).
  Most of the budget is the LSTM forward pass (~8 ms); tensor allocation and
  JSON serialisation account for the rest.
"""

import json
import logging
import os
import time
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from drift_monitor import DriftMonitor
from model import LSTMAutoencoder

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (override via environment variables at runtime)
# ---------------------------------------------------------------------------

MODEL_PATH     = os.getenv("MODEL_PATH",     "../models/lstm_ae_best.pth")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "../models/threshold.json")
INPUT_DIM      = int(os.getenv("INPUT_DIM",  "1"))
HIDDEN_DIM     = int(os.getenv("HIDDEN_DIM", "64"))
NUM_LAYERS     = int(os.getenv("NUM_LAYERS", "1"))
# WINDOW_SIZE=0 means "do not validate seq_len" (backwards-compatible default).
# Set to the training window size (e.g. 30) to reject windows of wrong length.
WINDOW_SIZE    = int(os.getenv("WINDOW_SIZE", "0"))
BASELINE_STATS_PATH = os.path.join(os.path.dirname(MODEL_PATH), "baseline_stats.json")
DRIFT_WINDOW_SIZE   = int(os.getenv("DRIFT_WINDOW_SIZE", "200"))

# Threshold loaded from saved file at startup; env var is a manual override only.
# Previously hardcoded to 0.05 while the actual trained threshold was ~6.99,
# meaning every window would have been flagged as anomalous.
_THRESHOLD_ENV_OVERRIDE = os.getenv("THRESHOLD")  # None if not set

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LSTM Autoencoder Anomaly Detection API",
    description="Detects anomalies in multivariate time-series windows using an LSTM Autoencoder trained on NASA SMAP satellite telemetry.",
    version="1.0.0",
)

# Global state populated at startup
_model: LSTMAutoencoder = None
_device: torch.device = None
_default_threshold: float = 0.05  # overwritten at startup from threshold.json
_drift_monitor: DriftMonitor = None


@app.on_event("startup")
def load_model():  # noqa: C901
    """Load model weights and deployment threshold once when the server starts.

    Threshold is read from the JSON file saved by train.py (99th percentile of
    training reconstruction errors).  This ensures the API uses the same
    threshold that was derived from the training distribution — not an
    arbitrary hardcoded value.  An explicit THRESHOLD env var still takes
    precedence for manual overrides.
    """
    global _model, _device, _default_threshold
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
        logger.info("Loaded model weights from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found at %s — serving with random weights", MODEL_PATH)

    _model.eval()

    # Initialise drift monitor from training baseline stats
    global _drift_monitor
    _drift_monitor = DriftMonitor.from_file(BASELINE_STATS_PATH, window_size=DRIFT_WINDOW_SIZE)

    # Load threshold — prefer saved file, fall back to env var
    if _THRESHOLD_ENV_OVERRIDE is not None:
        _default_threshold = float(_THRESHOLD_ENV_OVERRIDE)
        logger.info("Threshold set from env var: %s", _default_threshold)
    elif os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as fh:
            _default_threshold = float(json.load(fh)["threshold"])
        logger.info("Threshold loaded from %s: %.6f", THRESHOLD_PATH, _default_threshold)
    else:
        logger.warning(
            "Threshold file not found at %s — using fallback %.4f. "
            "Re-run train.py to generate threshold.json.",
            THRESHOLD_PATH,
            _default_threshold,
        )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class WindowRequest(BaseModel):
    """A single time-series window to score."""

    window: List[List[float]] = Field(
        ...,
        description=(
            "2-D list of shape [seq_len, num_features]. "
            "For SMAP channel E-7 this is [30, 25]."
        ),
    )
    threshold: float = Field(
        default=None,
        description=(
            "Anomaly score threshold above which is_anomaly=True. "
            "Defaults to the deployment threshold loaded from threshold.json."
        ),
    )

    def resolved_threshold(self) -> float:
        return self.threshold if self.threshold is not None else _default_threshold


class AnomalyResponse(BaseModel):
    """Anomaly detection result for a single window."""

    anomaly_score: float = Field(..., description="Mean-squared reconstruction error.")
    is_anomaly: bool = Field(..., description="True if anomaly_score >= threshold.")
    threshold_used: float = Field(..., description="Threshold applied to produce is_anomaly.")
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds.")
    drift_detected: Optional[bool] = Field(
        default=None,
        description=(
            "True if a KS test on the rolling reconstruction-error window detects "
            "a distribution shift from the training baseline. None if the window "
            "is not yet full (insufficient data for a reliable test)."
        ),
    )
    drift_p_value: Optional[float] = Field(
        default=None,
        description="KS test p-value. Values below 0.05 indicate statistically significant drift.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Ops"])
def health():
    """Readiness probe — returns 200 only when the server is fully ready.

    Checks that the model is loaded and a valid threshold is configured.
    Returns 503 if either condition is not met so load balancers and
    Kubernetes readiness probes do not route traffic to a broken instance.
    """
    if _model is None:
        logger.error("Health check failed: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _default_threshold <= 0:
        logger.error("Health check failed: threshold not configured (value=%.4f)", _default_threshold)
        raise HTTPException(status_code=503, detail="Threshold not configured")
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": _default_threshold,
        "device": str(_device),
    }


@app.get("/info", tags=["Ops"])
def info():
    """Return model configuration metadata."""
    return {
        "model": "LSTMAutoencoder",
        "dataset": "NASA SMAP",
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "default_threshold": _default_threshold,
        "threshold_source": THRESHOLD_PATH if os.path.exists(THRESHOLD_PATH) else "fallback",
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

    if WINDOW_SIZE > 0 and seq_len != WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Expected seq_len={WINDOW_SIZE} (model training window size), got {seq_len}.",
        )

    # --- Build tensor: (1, seq_len, features) ---
    x = torch.tensor(np.array(window, dtype=np.float32)).unsqueeze(0).to(_device)  # (1, W, F)

    # --- Inference ---
    with torch.no_grad():
        recon, _ = _model(x)
        mse = float(((recon - x) ** 2).mean().item())

    threshold = request.resolved_threshold()
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Update drift monitor and check distribution shift
    _drift_monitor.update(mse)
    drift_status = _drift_monitor.check()

    return AnomalyResponse(
        anomaly_score=round(mse, 6),
        is_anomaly=mse >= threshold,
        threshold_used=threshold,
        latency_ms=round(latency_ms, 3),
        drift_detected=drift_status.drift_detected if drift_status.p_value is not None else None,
        drift_p_value=drift_status.p_value,
    )


@app.get("/drift/status", tags=["Ops"])
def drift_status():
    """
    Return the current drift monitor state.

    Reports whether the rolling window of recent reconstruction errors has
    diverged from the training baseline distribution (KS test).  Use this
    endpoint to build operational dashboards or trigger retraining alerts.
    """
    status = _drift_monitor.check()
    return {
        "drift_detected": status.drift_detected,
        "p_value": status.p_value,
        "ks_statistic": status.ks_statistic,
        "mean_shift_sigmas": status.mean_shift_sigmas,
        "message": status.message,
        "monitor_stats": _drift_monitor.stats,
    }


@app.post("/predict/batch", response_model=List[AnomalyResponse], tags=["Inference"])
def predict_batch(requests: List[WindowRequest]) -> List[AnomalyResponse]:
    """Score a batch of time-series windows in a single forward pass.

    Stacks all windows into one tensor (B, W, F) and runs one LSTM forward
    pass — significantly faster than calling /predict N times in a loop
    (previously that was the implementation, making the batch endpoint a
    performance anti-pattern rather than an optimisation).
    """
    if not requests:
        return []

    t_start = time.perf_counter()

    # Validate shape for all windows before touching the model
    for i, req in enumerate(requests):
        if not req.window or not isinstance(req.window[0], list):
            raise HTTPException(status_code=400, detail=f"Request {i}: window must be a 2-D list.")
        n_features = len(req.window[0])
        if n_features != INPUT_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Request {i}: expected {INPUT_DIM} feature(s), got {n_features}.",
            )
        if WINDOW_SIZE > 0 and len(req.window) != WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Request {i}: expected seq_len={WINDOW_SIZE}, got {len(req.window)}.",
            )

    # Stack into a single batch tensor — requires all windows to have the same seq_len
    try:
        windows_np = np.stack([np.array(req.window, dtype=np.float32) for req in requests])
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"All windows must have the same seq_len for batch inference: {exc}",
        )

    x = torch.tensor(windows_np).to(_device)  # (B, W, F)

    # Single forward pass for the entire batch
    with torch.no_grad():
        recon, _ = _model(x)
        mses = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().tolist()  # (B,)

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0
    per_window_ms = round(total_latency_ms / len(requests), 3)

    # Feed all batch scores into the drift monitor, then do a single check
    for mse in mses:
        _drift_monitor.update(mse)
    drift_status = _drift_monitor.check()

    return [
        AnomalyResponse(
            anomaly_score=round(mse, 6),
            is_anomaly=mse >= req.resolved_threshold(),
            threshold_used=req.resolved_threshold(),
            latency_ms=per_window_ms,
            drift_detected=drift_status.drift_detected if drift_status.p_value is not None else None,
            drift_p_value=drift_status.p_value,
        )
        for mse, req in zip(mses, requests)
    ]
