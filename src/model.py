"""
model.py — LSTM Autoencoder for Time-Series Anomaly Detection

Architecture overview:

  Input (B, W, F)
      │
      ▼
  ┌─────────────────────────────────────┐
  │  Encoder LSTM  (F → hidden_dim)     │
  │  hidden state h_n  ← bottleneck    │
  └──────────────┬──────────────────────┘
                 │  repeat W times
                 ▼
  ┌─────────────────────────────────────┐
  │  Decoder LSTM  (hidden_dim → hid)   │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  Linear projection  (hid → F)       │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  Reconstruction (B, W, F)
                 │
                 ▼
         MSE vs original
                 │
                 ▼
  High MSE → anomaly flag

The model is trained on normal data only, so anomalous patterns produce a
high reconstruction error at inference time.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# LSTM Autoencoder
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence autoencoder built from two stacked LSTMs.

    LSTM gate recap (for reference):
        forget gate  (f_t): decides what to discard from the cell state
                           f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
        input  gate  (i_t): decides which new values to write to the cell state
                           i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
        output gate  (o_t): decides what to expose as the hidden state
                           o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
    These gates allow LSTMs to learn long-range dependencies while avoiding
    the vanishing-gradient problem that afflicts vanilla RNNs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Parameters
        ----------
        input_dim  : int   Number of features per time-step (F).
        hidden_dim : int   Size of LSTM hidden state (bottleneck width).
        num_layers : int   Number of stacked LSTM layers in encoder / decoder.
        dropout    : float Dropout probability between LSTM layers (0 if num_layers=1).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: maps (B, W, input_dim) → hidden state (num_layers, B, hidden_dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder: takes bottleneck repeated W times → reconstructed hidden seq
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project decoder outputs back to input feature space
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, W, F)

        Returns
        -------
        reconstruction : torch.Tensor, shape (B, W, F)
        bottleneck     : torch.Tensor, shape (B, hidden_dim)
                         Last-layer hidden state from the encoder — a compact
                         representation of the entire input window.
        """
        # --- Encode ---
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (num_layers, B, hidden_dim)
        bottleneck = h_n[-1]  # take the top (last) layer's hidden state → (B, hidden_dim)

        # --- Decode ---
        seq_len = x.size(1)
        # Repeat the bottleneck vector W times to seed the decoder at every step
        decoder_input = bottleneck.unsqueeze(1).repeat(1, seq_len, 1)  # (B, W, hidden_dim)
        decoder_out, _ = self.decoder(decoder_input)                    # (B, W, hidden_dim)

        # --- Project to input space ---
        reconstruction = self.output_layer(decoder_out)  # (B, W, F)

        return reconstruction, bottleneck


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def reconstruction_errors(
    model: LSTMAutoencoder,
    loader,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-window MSE reconstruction error.

    The MSE is averaged over both the sequence-length and feature dimensions
    so the returned score is a scalar per window, independent of window size
    and number of features.

    Parameters
    ----------
    model  : LSTMAutoencoder
    loader : DataLoader (shuffle=False to preserve temporal order)
    device : torch.device

    Returns
    -------
    errors : torch.Tensor, shape (N_windows,)
             Higher values indicate more anomalous windows.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, _ = model(x)
            # MSE per window: mean over (W, F) dimensions
            mse = ((recon - x) ** 2).mean(dim=(1, 2))  # (B,)
            all_errors.append(mse.cpu())

    return torch.cat(all_errors)  # (N_windows,)


def isolation_forest_errors(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
) -> np.ndarray:
    """Baseline anomaly scores using Isolation Forest.

    Flattens each window from (W, F) to (W*F,) and fits an Isolation Forest on
    the training windows.  Returns negated ``score_samples`` on the test set so
    that higher values correspond to more anomalous windows (consistent with the
    MSE convention used by the LSTM-AE).

    Parameters
    ----------
    train_windows : np.ndarray, shape (N_train, W, F)
    test_windows  : np.ndarray, shape (N_test,  W, F)

    Returns
    -------
    scores : np.ndarray, shape (N_test,)
             Higher → more anomalous.
    """
    N_train, W, F = train_windows.shape
    N_test = test_windows.shape[0]

    X_train = train_windows.reshape(N_train, W * F)
    X_test = test_windows.reshape(N_test, W * F)

    clf = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train)

    # score_samples returns higher values for inliers; negate so anomalies score high
    scores = -clf.score_samples(X_test)
    return scores
