"""
Shared pytest fixtures.

Fixtures defined here are available to all test files automatically.
Scope notes:
  - "function" (default) — fresh copy per test, safest
  - "module" — one copy for the whole test file, used for expensive setup
               like building a real model or starting the API client
"""

import json

import numpy as np
import pytest
import torch

from model import LSTMAutoencoder


# ---------------------------------------------------------------------------
# Raw data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def train_data(rng):
    """Small (200, 5) float32 array — stands in for a single SMAP channel."""
    return rng.standard_normal((200, 5)).astype(np.float32)


@pytest.fixture
def test_data(rng):
    return rng.standard_normal((80, 5)).astype(np.float32)


@pytest.fixture
def binary_labels(rng):
    """Binary labels with ~10% anomaly rate, length matching test_data."""
    labels = np.zeros(80, dtype=int)
    labels[30:40] = 1   # one contiguous anomaly segment
    return labels


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    """Tiny LSTMAutoencoder (input_dim=5, hidden=16) for fast forward-pass tests."""
    return LSTMAutoencoder(input_dim=5, hidden_dim=16, num_layers=1, dropout=0.0)


@pytest.fixture
def model_checkpoint(tmp_path, small_model):
    """Saves small_model to a temp .pth file; returns the path."""
    path = tmp_path / "model.pth"
    torch.save(small_model.state_dict(), path)
    return str(path)


@pytest.fixture
def threshold_file(tmp_path):
    """Writes a threshold.json with value 0.5; returns the path."""
    path = tmp_path / "threshold.json"
    with open(path, "w") as fh:
        json.dump({"threshold": 0.5}, fh)
    return str(path)


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """
    A FastAPI TestClient backed by a real (tiny) LSTMAutoencoder.

    Uses INPUT_DIM=5 to match the other fixtures.  The model weights and
    threshold are written to a temp directory and the api module globals are
    patched before the startup event runs, so load_model() picks up the
    correct paths.
    """
    import api

    tmp = tmp_path_factory.mktemp("api_models")
    model_path = str(tmp / "model.pth")
    threshold_path = str(tmp / "threshold.json")

    m = LSTMAutoencoder(input_dim=5, hidden_dim=16, num_layers=1, dropout=0.0)
    torch.save(m.state_dict(), model_path)
    with open(threshold_path, "w") as fh:
        json.dump({"threshold": 0.5}, fh)

    # Patch module globals BEFORE TestClient triggers the startup event.
    # load_model() reads these at runtime, not at import time.
    api.MODEL_PATH = model_path
    api.THRESHOLD_PATH = threshold_path
    api.INPUT_DIM = 5
    api.HIDDEN_DIM = 16
    api.NUM_LAYERS = 1
    api._THRESHOLD_ENV_OVERRIDE = None

    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        yield client
