"""
dataset.py — NASA SMAP Time-Series Dataset Utilities

WHY we window:
  LSTMs process sequences of fixed length. Raw telemetry is a 1-D signal of
  arbitrary length; by sliding a window of size W we convert it into N overlapping
  sub-sequences, each of shape (W, F).  This gives the model local temporal context
  and also lets us map anomaly scores back to individual time-steps.

WHY we train only on normal data:
  The autoencoder is trained to minimise reconstruction error on healthy signals.
  When it later sees anomalous patterns it has never encountered, the decoder fails
  to reconstruct them accurately, producing a high MSE.  If we trained on anomalies
  too, the model would learn to reconstruct them well and the signal would be lost.

WHY StandardScaler is fit only on train:
  Fitting the scaler on the test set would constitute data leakage — the model
  would indirectly "see" test statistics during training.  We compute mean/std on
  the training split only and apply the same transform to the test split.
"""

import ast
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

WINDOW_SIZE = 128
STRIDE = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_channel(channel: str, data_dir: str = "../data"):
    """Load train and test .npy files for a single SMAP channel.

    Fits StandardScaler on train only to avoid leakage, then transforms both
    splits.

    Parameters
    ----------
    channel : str
        Channel identifier, e.g. ``"P-1"``.
    data_dir : str
        Root directory containing ``train/`` and ``test/`` sub-folders.

    Returns
    -------
    train_data : np.ndarray, shape (T_train, F)
    test_data  : np.ndarray, shape (T_test,  F)
    scaler     : fitted StandardScaler (kept for inverse-transform if needed)
    """
    train_path = os.path.join(data_dir, "train", f"{channel}.npy")
    test_path = os.path.join(data_dir, "test", f"{channel}.npy")

    train_data = np.load(train_path).astype(np.float32)
    test_data = np.load(test_path).astype(np.float32)

    # Reshape 1-D signals to (T, 1) so scaler API is consistent
    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)
    if test_data.ndim == 1:
        test_data = test_data.reshape(-1, 1)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)   # fit + transform on train
    test_data = scaler.transform(test_data)          # transform only on test

    return train_data, test_data, scaler


def load_labels(channel: str, test_length: int, data_dir: str = "../data") -> np.ndarray:
    """Build a binary anomaly label array for the test split.

    Reads ``labeled_anomalies.csv`` which ships with the SMAP dataset and
    contains columns ``chan_id`` and ``anomaly_sequences``.
    ``anomaly_sequences`` is a string representation of a list of [start, end]
    pairs, e.g. ``"[[100, 200], [450, 500]]"``.

    Parameters
    ----------
    channel     : str   Channel identifier, e.g. ``"P-1"``.
    test_length : int   Length of the test array (used to size the label vector).
    data_dir    : str   Root directory containing ``labeled_anomalies.csv``.

    Returns
    -------
    labels : np.ndarray of dtype int, shape (test_length,)
             1 where anomalous, 0 elsewhere.
    """
    csv_path = os.path.join(data_dir, "labeled_anomalies.csv")
    df = pd.read_csv(csv_path)
    row = df[df["chan_id"] == channel]

    labels = np.zeros(test_length, dtype=int)

    if row.empty:
        return labels

    sequences_str = row["anomaly_sequences"].values[0]
    # ast.literal_eval safely parses "[[100, 200], [450, 500]]" → list of lists
    sequences = ast.literal_eval(sequences_str)

    for start, end in sequences:
        labels[int(start): int(end) + 1] = 1

    return labels


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(data: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    """Create overlapping sliding windows from a time series.

    Parameters
    ----------
    data        : np.ndarray, shape (T, F)
    window_size : int   Number of time-steps per window.
    stride      : int   Step between consecutive windows.  stride=1 maximises
                        the number of training samples from a limited dataset.

    Returns
    -------
    windows : np.ndarray, shape (N, window_size, F)
    """
    T, F = data.shape
    n_windows = (T - window_size) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size, F))
    # sliding_window_view returns shape (n_windows, 1, window_size, F) — squeeze axis 1
    windows = windows[:n_windows:stride].reshape(n_windows, window_size, F)
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """Wraps a numpy array of windows for use with DataLoader.

    ``__getitem__`` returns ``(window, window)`` — input equals target — because
    we are training a reconstruction autoencoder.  High reconstruction error on
    the test set signals anomalies.
    """

    def __init__(self, windows: np.ndarray):
        """
        Parameters
        ----------
        windows : np.ndarray, shape (N, W, F)
        """
        import torch
        self.windows = torch.from_numpy(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return w, w   # (input, target) — reconstruction objective


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    train_data: np.ndarray,
    test_data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    batch_size: int = 64,
):
    """Build DataLoaders and return raw window arrays for scoring.

    Parameters
    ----------
    train_data  : np.ndarray, shape (T_train, F)
    test_data   : np.ndarray, shape (T_test,  F)
    window_size : int
    batch_size  : int

    Returns
    -------
    train_loader  : DataLoader  (shuffle=True)
    test_loader   : DataLoader  (shuffle=False, preserves temporal order)
    train_windows : np.ndarray, shape (N_train, W, F)
    test_windows  : np.ndarray, shape (N_test,  W, F)
    """
    train_windows = make_windows(train_data, window_size, stride=STRIDE)
    test_windows = make_windows(test_data, window_size, stride=1)

    train_ds = TimeSeriesDataset(train_windows)
    test_ds = TimeSeriesDataset(test_windows)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_windows, test_windows
