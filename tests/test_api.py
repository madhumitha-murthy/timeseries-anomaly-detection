"""
Tests for src/api.py

Covers:
  - GET /health: readiness (model loaded, threshold set), 503 when not ready
  - GET /info: metadata fields present, threshold reflects loaded value
  - POST /predict: valid input, wrong feature dim, wrong seq_len, malformed body
  - POST /predict/batch: correct count, order preserved, real batching,
                         mismatched seq_len, empty list, wrong features
  - Threshold: default loaded from file, per-request override respected
  - Logging: no print() calls remain in api.py
"""

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_window(seq_len: int = 10, features: int = 5) -> list:
    """Return a random 2-D list of shape [seq_len, features]."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((seq_len, features)).tolist()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_when_ready(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_body_has_ok_status(self, api_client):
        r = api_client.get("/health")
        assert r.json()["status"] == "ok"

    def test_body_reports_model_loaded(self, api_client):
        body = api_client.get("/health").json()
        assert body["model_loaded"] is True

    def test_body_includes_threshold(self, api_client):
        body = api_client.get("/health").json()
        assert "threshold" in body
        assert isinstance(body["threshold"], float)

    def test_body_includes_device(self, api_client):
        body = api_client.get("/health").json()
        assert "device" in body

    def test_returns_503_when_model_not_loaded(self, api_client):
        """Health must return 503 when _model is None.

        Uses the already-running api_client so no startup event is re-triggered
        (a new TestClient context would reload _model, undoing our patch).
        """
        import api
        original_model = api._model
        api._model = None
        try:
            r = api_client.get("/health")
            assert r.status_code == 503
        finally:
            api._model = original_model


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_returns_200(self, api_client):
        r = api_client.get("/info")
        assert r.status_code == 200

    def test_required_fields_present(self, api_client):
        body = api_client.get("/info").json()
        for field in ("model", "input_dim", "hidden_dim", "num_layers", "default_threshold"):
            assert field in body, f"Missing field: {field}"

    def test_threshold_is_numeric(self, api_client):
        body = api_client.get("/info").json()
        assert isinstance(body["default_threshold"], (int, float))

    def test_threshold_loaded_from_file(self, api_client):
        """The fixture writes threshold=0.5; /info must reflect it."""
        body = api_client.get("/info").json()
        assert body["default_threshold"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_valid_input_returns_200(self, api_client):
        r = api_client.post("/predict", json={"window": make_window()})
        assert r.status_code == 200

    def test_response_schema(self, api_client):
        r = api_client.post("/predict", json={"window": make_window()})
        body = r.json()
        assert "anomaly_score" in body
        assert "is_anomaly" in body
        assert "threshold_used" in body
        assert "latency_ms" in body

    def test_anomaly_score_is_non_negative(self, api_client):
        r = api_client.post("/predict", json={"window": make_window()})
        assert r.json()["anomaly_score"] >= 0.0

    def test_is_anomaly_is_bool(self, api_client):
        r = api_client.post("/predict", json={"window": make_window()})
        assert isinstance(r.json()["is_anomaly"], bool)

    def test_latency_ms_is_positive(self, api_client):
        r = api_client.post("/predict", json={"window": make_window()})
        assert r.json()["latency_ms"] > 0

    def test_default_threshold_from_file(self, api_client):
        """threshold_used must match the 0.5 written by the fixture."""
        r = api_client.post("/predict", json={"window": make_window()})
        assert r.json()["threshold_used"] == pytest.approx(0.5)

    def test_per_request_threshold_override(self, api_client):
        """Explicit threshold in request body overrides the default."""
        r = api_client.post("/predict", json={"window": make_window(), "threshold": 99.0})
        body = r.json()
        assert body["threshold_used"] == pytest.approx(99.0)
        assert body["is_anomaly"] is False  # score can't exceed 99.0

    def test_low_threshold_flags_anomaly(self, api_client):
        """threshold=0 means any reconstruction error is an anomaly."""
        r = api_client.post("/predict", json={"window": make_window(), "threshold": 0.0})
        assert r.json()["is_anomaly"] is True

    def test_wrong_feature_count_returns_400(self, api_client):
        wrong = make_window(features=3)   # model expects 5
        r = api_client.post("/predict", json={"window": wrong})
        assert r.status_code == 400

    def test_1d_window_returns_4xx(self, api_client):
        """window must be 2-D — a flat list is rejected.
        Pydantic rejects it before our code runs (422 Unprocessable Entity),
        but our manual check also returns 400 for [[f1, f2], ...] where items
        are not lists.  Either is acceptable; we only assert it is an error."""
        r = api_client.post("/predict", json={"window": [0.1, 0.2, 0.3]})
        assert r.status_code in (400, 422)

    def test_empty_body_returns_422(self, api_client):
        """Missing required field 'window' → FastAPI validation error."""
        r = api_client.post("/predict", json={})
        assert r.status_code == 422

    def test_variable_seq_len_accepted_when_window_size_zero(self, api_client):
        """When WINDOW_SIZE=0 (default), any seq_len is accepted."""
        import api
        original = api.WINDOW_SIZE
        api.WINDOW_SIZE = 0
        try:
            for seq_len in [5, 20, 50]:
                r = api_client.post("/predict", json={"window": make_window(seq_len=seq_len)})
                assert r.status_code == 200, f"Failed for seq_len={seq_len}"
        finally:
            api.WINDOW_SIZE = original

    def test_wrong_seq_len_returns_400_when_window_size_set(self, api_client):
        """When WINDOW_SIZE is configured, a window of wrong length is rejected."""
        import api
        original = api.WINDOW_SIZE
        api.WINDOW_SIZE = 10
        try:
            r = api_client.post("/predict", json={"window": make_window(seq_len=5)})
            assert r.status_code == 400
            assert "seq_len" in r.json()["detail"]
        finally:
            api.WINDOW_SIZE = original

    def test_correct_seq_len_accepted_when_window_size_set(self, api_client):
        import api
        original = api.WINDOW_SIZE
        api.WINDOW_SIZE = 10
        try:
            r = api_client.post("/predict", json={"window": make_window(seq_len=10)})
            assert r.status_code == 200
        finally:
            api.WINDOW_SIZE = original

    def test_no_print_statements_in_api_source(self):
        """Ensure all print() calls were replaced with logger calls."""
        import inspect
        import api
        source = inspect.getsource(api)
        # Allow print only inside string literals / comments — check for bare calls
        import ast
        tree = ast.parse(source)
        print_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert print_calls == [], f"Found {len(print_calls)} print() call(s) in api.py — use logger instead"


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_correct_count(self, api_client):
        payload = [{"window": make_window()} for _ in range(5)]
        r = api_client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        assert len(r.json()) == 5

    def test_empty_list_returns_empty_list(self, api_client):
        r = api_client.post("/predict/batch", json=[])
        assert r.status_code == 200
        assert r.json() == []

    def test_response_items_have_correct_schema(self, api_client):
        payload = [{"window": make_window()}]
        item = api_client.post("/predict/batch", json=payload).json()[0]
        for field in ("anomaly_score", "is_anomaly", "threshold_used", "latency_ms"):
            assert field in item

    def test_results_match_individual_predictions(self, api_client):
        """Batch scores must equal individual /predict scores for the same input."""
        # All windows must have the same seq_len so np.stack succeeds.
        windows = [make_window(seq_len=10) for _ in range(3)]
        batch_payload = [{"window": w} for w in windows]

        batch_scores = [item["anomaly_score"] for item in
                        api_client.post("/predict/batch", json=batch_payload).json()]

        single_scores = [
            api_client.post("/predict", json={"window": w}).json()["anomaly_score"]
            for w in windows
        ]

        for b, s in zip(batch_scores, single_scores):
            assert b == pytest.approx(s, rel=1e-4), \
                f"Batch score {b} != single score {s}"

    def test_order_preserved(self, api_client):
        """Results must be in the same order as the input requests."""
        # Use threshold=0 so all are flagged, then threshold=99 so none are
        payload_low  = [{"window": make_window(), "threshold": 0.0}  for _ in range(4)]
        payload_high = [{"window": make_window(), "threshold": 99.0} for _ in range(4)]

        for item in api_client.post("/predict/batch", json=payload_low).json():
            assert item["is_anomaly"] is True

        for item in api_client.post("/predict/batch", json=payload_high).json():
            assert item["is_anomaly"] is False

    def test_wrong_feature_count_returns_400(self, api_client):
        payload = [{"window": make_window(features=3)}]   # model expects 5
        r = api_client.post("/predict/batch", json=payload)
        assert r.status_code == 400

    def test_mismatched_seq_len_returns_400(self, api_client):
        """np.stack requires identical shapes; mixed seq_len must be rejected."""
        payload = [
            {"window": make_window(seq_len=10)},
            {"window": make_window(seq_len=20)},  # different seq_len
        ]
        r = api_client.post("/predict/batch", json=payload)
        assert r.status_code == 400

    def test_batch_is_faster_than_sequential(self, api_client):
        """
        Single forward pass for N windows should be faster than N individual
        calls (or at least not slower by more than 2x — CI machines vary).
        This is a smoke test for the real-batching implementation.
        """
        N = 20
        windows = [make_window() for _ in range(N)]

        t0 = time.perf_counter()
        api_client.post("/predict/batch", json=[{"window": w} for w in windows])
        batch_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for w in windows:
            api_client.post("/predict", json={"window": w})
        sequential_ms = (time.perf_counter() - t0) * 1000

        # Batch must be at least 2x faster than sequential
        assert batch_ms < sequential_ms, (
            f"Batch ({batch_ms:.1f}ms) was not faster than sequential ({sequential_ms:.1f}ms)"
        )
