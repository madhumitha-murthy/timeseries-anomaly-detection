"""
Tests for src/model.py

Covers:
  - LSTMAutoencoder forward pass: output shapes, bottleneck shape
  - LSTMAutoencoder eval mode: no gradient computation
  - reconstruction_errors: count, non-negativity, higher on unseen data
  - isolation_forest_errors: shape, contamination clipping
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMAutoencoder, isolation_forest_errors, reconstruction_errors


# ---------------------------------------------------------------------------
# LSTMAutoencoder — forward pass
# ---------------------------------------------------------------------------

class TestLSTMAutoencoderForward:
    def test_reconstruction_shape(self, small_model):
        x = torch.randn(4, 10, 5)   # (B=4, W=10, F=5)
        recon, _ = small_model(x)
        assert recon.shape == (4, 10, 5)

    def test_bottleneck_shape(self, small_model):
        x = torch.randn(4, 10, 5)
        _, bottleneck = small_model(x)
        assert bottleneck.shape == (4, 16)  # (B, hidden_dim)

    def test_single_sample(self, small_model):
        x = torch.randn(1, 30, 5)
        recon, bottleneck = small_model(x)
        assert recon.shape == (1, 30, 5)
        assert bottleneck.shape == (1, 16)

    def test_output_is_float32(self, small_model):
        x = torch.randn(2, 10, 5)
        recon, _ = small_model(x)
        assert recon.dtype == torch.float32

    def test_reconstruction_differs_from_input(self, small_model):
        """Untrained model should not perfectly reconstruct its input."""
        x = torch.randn(4, 10, 5)
        recon, _ = small_model(x)
        assert not torch.allclose(recon, x)

    def test_different_sequence_lengths(self, small_model):
        """LSTM should handle any seq_len at inference."""
        for seq_len in [5, 15, 50]:
            x = torch.randn(2, seq_len, 5)
            recon, _ = small_model(x)
            assert recon.shape == (2, seq_len, 5)

    def test_no_gradients_in_eval_mode(self, small_model):
        small_model.eval()
        x = torch.randn(2, 10, 5)
        with torch.no_grad():
            recon, _ = small_model(x)
        assert not recon.requires_grad

    def test_multilayer_model(self):
        model = LSTMAutoencoder(input_dim=3, hidden_dim=8, num_layers=2, dropout=0.2)
        x = torch.randn(4, 10, 3)
        recon, bottleneck = model(x)
        assert recon.shape == (4, 10, 3)
        assert bottleneck.shape == (4, 8)


# ---------------------------------------------------------------------------
# reconstruction_errors
# ---------------------------------------------------------------------------

class TestReconstructionErrors:
    def _make_loader(self, n_windows=50, window_size=10, features=5, batch_size=16):
        windows = torch.randn(n_windows, window_size, features)
        ds = TensorDataset(windows, windows)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_returns_one_score_per_window(self, small_model):
        loader = self._make_loader(n_windows=50)
        errors = reconstruction_errors(small_model, loader, torch.device("cpu"))
        assert errors.shape == (50,)

    def test_errors_are_non_negative(self, small_model):
        """MSE is always >= 0."""
        loader = self._make_loader(n_windows=30)
        errors = reconstruction_errors(small_model, loader, torch.device("cpu"))
        assert (errors >= 0).all()

    def test_perfect_reconstruction_gives_zero_error(self):
        """If recon == input, MSE == 0."""
        # Build a model that reconstructs perfectly by making recon = input directly
        class IdentityAE(torch.nn.Module):
            def forward(self, x):
                return x, x[:, 0, :]  # recon = input, bottleneck = first timestep

        model = IdentityAE()
        windows = torch.randn(20, 10, 5)
        loader = DataLoader(TensorDataset(windows, windows), batch_size=8)
        errors = reconstruction_errors(model, loader, torch.device("cpu"))
        assert torch.allclose(errors, torch.zeros(20), atol=1e-6)

    def test_model_set_to_eval_after_call(self, small_model):
        loader = self._make_loader(n_windows=20)
        small_model.train()
        reconstruction_errors(small_model, loader, torch.device("cpu"))
        assert not small_model.training


# ---------------------------------------------------------------------------
# isolation_forest_errors
# ---------------------------------------------------------------------------

class TestIsolationForestErrors:
    def test_output_shape(self):
        train = np.random.randn(100, 10, 3).astype(np.float32)
        test  = np.random.randn(40,  10, 3).astype(np.float32)
        scores = isolation_forest_errors(train, test, contamination=0.1)
        assert scores.shape == (40,)

    def test_output_is_float(self):
        train = np.random.randn(80, 5, 2).astype(np.float32)
        test  = np.random.randn(20, 5, 2).astype(np.float32)
        scores = isolation_forest_errors(train, test, contamination=0.1)
        assert scores.dtype in (np.float32, np.float64)

    def test_contamination_zero_clipped(self):
        """contamination=0 should be clipped to 1e-3, not raise."""
        train = np.random.randn(100, 5, 2).astype(np.float32)
        test  = np.random.randn(20,  5, 2).astype(np.float32)
        scores = isolation_forest_errors(train, test, contamination=0.0)
        assert scores.shape == (20,)

    def test_contamination_over_half_clipped(self):
        """contamination=0.9 should be clipped to 0.5, not raise."""
        train = np.random.randn(100, 5, 2).astype(np.float32)
        test  = np.random.randn(20,  5, 2).astype(np.float32)
        scores = isolation_forest_errors(train, test, contamination=0.9)
        assert scores.shape == (20,)

    def test_anomalous_windows_score_higher(self):
        """Outlier windows should receive higher anomaly scores than inlier windows."""
        rng = np.random.default_rng(0)

        # Train on clean Gaussian data
        train = rng.standard_normal((200, 10, 1)).astype(np.float32)

        # Mix: 50 clean + 50 obvious outliers (signal = 100)
        clean   = rng.standard_normal((50, 10, 1)).astype(np.float32)
        outlier = np.full((50, 10, 1), 100.0, dtype=np.float32)
        test    = np.concatenate([clean, outlier], axis=0)

        scores = isolation_forest_errors(train, test, contamination=0.5)
        clean_mean   = scores[:50].mean()
        outlier_mean = scores[50:].mean()
        assert outlier_mean > clean_mean
