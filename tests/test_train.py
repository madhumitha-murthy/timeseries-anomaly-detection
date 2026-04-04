"""
Tests for src/train.py utilities

Covers:
  - window_to_point_scores: shape, max-pooling correctness, edge cases
  - find_best_threshold: perfect separation, returns valid range, all-zero edge case
  - EarlyStopping: patience, reset on improvement, min_delta
  - train_epoch / val_epoch: loss is finite, val doesn't update parameters
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from train import EarlyStopping, find_best_threshold, train_epoch, val_epoch, window_to_point_scores


# ---------------------------------------------------------------------------
# window_to_point_scores
# ---------------------------------------------------------------------------

class TestWindowToPointScores:
    def test_output_shape(self):
        scores = np.array([1.0, 2.0, 3.0])
        result = window_to_point_scores(scores, total_length=10, window_size=3)
        assert result.shape == (10,)

    def test_max_pooling_correctness(self):
        """
        3 windows of size 3, total_length=5:
          window 0 covers [0,1,2] with score 1.0
          window 1 covers [1,2,3] with score 3.0
          window 2 covers [2,3,4] with score 2.0

        Expected point scores:
          t=0: max(1.0)       = 1.0
          t=1: max(1.0, 3.0)  = 3.0
          t=2: max(1.0, 3.0, 2.0) = 3.0
          t=3: max(3.0, 2.0)  = 3.0
          t=4: max(2.0)       = 2.0
        """
        scores = np.array([1.0, 3.0, 2.0])
        result = window_to_point_scores(scores, total_length=5, window_size=3)
        np.testing.assert_array_almost_equal(result, [1.0, 3.0, 3.0, 3.0, 2.0])

    def test_monotone_scores_propagate_correctly(self):
        scores = np.array([1.0, 2.0, 3.0])
        result = window_to_point_scores(scores, total_length=5, window_size=3)
        # Every point covered by window 2 (score=3.0) should be at least 3.0
        assert result[2] == pytest.approx(3.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(3.0)

    def test_output_is_float32(self):
        scores = np.array([0.1, 0.2], dtype=np.float32)
        result = window_to_point_scores(scores, total_length=4, window_size=3)
        assert result.dtype == np.float32

    def test_zero_scores(self):
        scores = np.zeros(5)
        result = window_to_point_scores(scores, total_length=7, window_size=3)
        assert (result == 0).all()

    def test_single_window(self):
        scores = np.array([0.9])
        result = window_to_point_scores(scores, total_length=5, window_size=3)
        # Window 0 covers [0,1,2]; points 3 and 4 are uncovered (remain 0)
        assert result[0] == pytest.approx(0.9)
        assert result[1] == pytest.approx(0.9)
        assert result[2] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# find_best_threshold
# ---------------------------------------------------------------------------

class TestFindBestThreshold:
    def test_perfect_separation_gives_f1_1(self):
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        labels = np.array([0,   0,   0,   1,   1])
        _, f1, preds = find_best_threshold(scores, labels)
        assert f1 == pytest.approx(1.0)
        np.testing.assert_array_equal(preds, labels)

    def test_threshold_within_score_range(self):
        rng = np.random.default_rng(42)
        scores = rng.random(100)
        labels = (scores > 0.7).astype(int)
        thresh, f1, _ = find_best_threshold(scores, labels)
        assert scores.min() <= thresh <= scores.max()

    def test_correlated_scores_give_positive_f1(self):
        rng = np.random.default_rng(7)
        scores = rng.random(200)
        labels = (scores > 0.6).astype(int)
        _, f1, _ = find_best_threshold(scores, labels)
        assert f1 > 0.0

    def test_all_zero_scores_returns_zero_f1(self):
        """No signal → can't beat zero F1."""
        scores = np.zeros(50)
        labels = np.zeros(50, dtype=int)
        _, f1, _ = find_best_threshold(scores, labels)
        assert f1 == pytest.approx(0.0)

    def test_predictions_are_binary(self):
        rng = np.random.default_rng(1)
        scores = rng.random(60)
        labels = (scores > 0.5).astype(int)
        _, _, preds = find_best_threshold(scores, labels)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predictions_length_matches_labels(self):
        scores = np.random.rand(80)
        labels = np.random.randint(0, 2, 80)
        _, _, preds = find_best_threshold(scores, labels)
        assert len(preds) == len(labels)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=3)
        assert not es.step(1.0)   # improvement → counter=0
        assert not es.step(0.9)   # improvement → counter=0
        assert not es.step(0.9)   # no improvement → counter=1
        assert not es.step(0.9)   # no improvement → counter=2
        assert     es.step(0.9)   # no improvement → counter=3 → STOP

    def test_does_not_stop_with_steady_improvement(self):
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            assert not es.step(loss)

    def test_resets_counter_on_improvement(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)    # improvement
        es.step(1.0)    # counter=1
        es.step(0.5)    # improvement → counter resets to 0
        assert not es.step(0.5)   # counter=1
        assert es.step(0.5)       # counter=2 → STOP (not 3, because reset happened)

    def test_min_delta_prevents_tiny_improvements_counting(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)    # sets best_loss=1.0
        # 0.95 < 1.0 but improvement < min_delta → treated as no improvement
        assert not es.step(0.95)  # counter=1
        assert es.step(0.95)      # counter=2 → STOP

    def test_min_delta_allows_large_improvements(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        assert not es.step(1.0)
        assert not es.step(0.5)   # improvement > min_delta → counter resets

    def test_best_loss_tracked_correctly(self):
        es = EarlyStopping(patience=5)
        es.step(1.0)
        es.step(0.8)
        es.step(0.3)
        es.step(0.5)    # worse — no update
        assert es.best_loss == pytest.approx(0.3)

    def test_initial_best_loss_is_inf(self):
        es = EarlyStopping(patience=3)
        assert es.best_loss == float("inf")


# ---------------------------------------------------------------------------
# train_epoch / val_epoch (integration — verifies training loop contract)
# ---------------------------------------------------------------------------

class TestTrainValEpoch:
    """Use a tiny linear autoencoder (not LSTM) for speed — we're testing
    the loop logic, not the architecture."""

    @pytest.fixture
    def tiny_setup(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10, 4),
            nn.ReLU(),
            nn.Linear(4, 10),
            nn.Unflatten(1, (5, 2)),
        )
        # Wrap to match (recon, bottleneck) return convention expected by train_epoch
        class WrappedModel(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x), x[:, 0, :]  # bottleneck = first step (dummy)

        wrapped = WrappedModel(model)
        windows = torch.randn(64, 5, 2)
        loader = DataLoader(TensorDataset(windows, windows), batch_size=16, shuffle=True, drop_last=True)
        optimizer = optim.Adam(wrapped.parameters(), lr=1e-2)
        criterion = nn.MSELoss()
        return wrapped, loader, optimizer, criterion

    def test_train_epoch_returns_finite_loss(self, tiny_setup):
        model, loader, optimizer, criterion = tiny_setup
        loss = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        assert np.isfinite(loss)

    def test_train_epoch_loss_decreases_over_multiple_epochs(self, tiny_setup):
        model, loader, optimizer, criterion = tiny_setup
        loss1 = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        loss10 = loss1
        for _ in range(10):
            loss10 = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        # After 12 epochs on simple data, loss should be lower than initial
        assert loss10 < loss1

    def test_val_epoch_does_not_update_parameters(self, tiny_setup):
        model, loader, optimizer, criterion = tiny_setup
        params_before = [p.clone() for p in model.parameters()]
        val_epoch(model, loader, criterion, torch.device("cpu"))
        params_after = list(model.parameters())
        for before, after in zip(params_before, params_after):
            assert torch.equal(before, after), "val_epoch must not modify model parameters"

    def test_val_epoch_returns_finite_loss(self, tiny_setup):
        model, loader, optimizer, criterion = tiny_setup
        loss = val_epoch(model, loader, criterion, torch.device("cpu"))
        assert np.isfinite(loss)
