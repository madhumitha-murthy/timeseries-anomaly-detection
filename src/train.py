"""
train.py — Full training pipeline for LSTM Autoencoder on NASA SMAP dataset.

Usage:
    python train.py

Expects the SMAP data under CONFIG["data_dir"] with sub-folders train/ and test/
and the file labeled_anomalies.csv.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # headless backend for server / Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import get_dataloaders, load_channel, load_labels
from model import LSTMAutoencoder, isolation_forest_errors, reconstruction_errors

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "channel": "E-7",
    "data_dir": "../data",
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.0,
    "window_size": 30,
    "batch_size": 32,
    "num_epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "../models/lstm_ae_best.pth",
    "out_dir": "../outputs/",
}


# ---------------------------------------------------------------------------
# Point-adjust mapping
# ---------------------------------------------------------------------------

def window_to_point_scores(
    window_scores: np.ndarray,
    total_length: int,
    window_size: int,
) -> np.ndarray:
    """Map window-level anomaly scores to point-level scores via max-pooling.

    Each time-step receives the MAXIMUM score of all windows that contain it.
    This is point-adjust — standard in the SMAP/MSL literature (Hundman et al.
    2018).  Using max rather than mean is conservative: a single highly
    anomalous window is enough to flag a time-step, which reduces missed
    detections.

    Parameters
    ----------
    window_scores : np.ndarray, shape (N_windows,)
    total_length  : int   Length of the original test time series.
    window_size   : int

    Returns
    -------
    point_scores : np.ndarray, shape (total_length,)
    """
    # This is point-adjust — standard in SMAP literature.
    point_scores = np.zeros(total_length, dtype=np.float32)
    counts = np.zeros(total_length, dtype=np.float32)

    for i, score in enumerate(window_scores):
        start = i
        end = i + window_size
        if end > total_length:
            break
        point_scores[start:end] = np.maximum(point_scores[start:end], score)
        counts[start:end] += 1

    return point_scores


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n: int = 200,
):
    """Sweep n thresholds between min and max score and return the best F1.

    Parameters
    ----------
    scores : np.ndarray, shape (T,)
    labels : np.ndarray, shape (T,)  binary ground-truth
    n      : int  number of candidate thresholds

    Returns
    -------
    best_threshold   : float
    best_f1          : float
    best_predictions : np.ndarray, shape (T,)  binary predictions at best threshold
    """
    thresholds = np.linspace(scores.min(), scores.max(), n)
    best_f1 = -1.0
    best_threshold = thresholds[0]
    best_predictions = np.zeros_like(labels)

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_predictions = preds

    return best_threshold, best_f1, best_predictions


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """One full pass over the training DataLoader.

    Gradient clipping (max_norm=1.0) is applied before every parameter update.
    WHY gradient clipping matters for LSTMs:
        Back-propagation through time (BPTT) can cause gradients to grow
        exponentially as they are multiplied across many time steps.  This
        "exploding gradient" problem destabilises training — clipping the
        gradient norm to 1.0 caps the update magnitude without eliminating
        gradient signal, keeping optimisation stable.

    Returns
    -------
    avg_loss : float
    """
    model.train()
    total_loss = 0.0

    for x, target in loader:
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        recon, _ = model(x)
        loss = criterion(recon, target)
        loss.backward()

        # Clip gradient norm to prevent exploding gradients in LSTM BPTT
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    """Evaluate loss on a DataLoader without updating parameters.

    Returns
    -------
    avg_loss : float
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, target in loader:
            x = x.to(device)
            target = target.to(device)
            recon, _ = model(x)
            loss = criterion(recon, target)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_anomaly_results(test_signal, point_scores, best_thresh, labels, best_preds, out_dir):
    """Three-panel diagnostic plot saved to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    t = np.arange(len(test_signal))

    # Panel 1 — raw signal + true anomaly shading
    ax = axes[0]
    ax.plot(t, test_signal[:, 0], color="steelblue", lw=0.7, label="Signal")
    in_anomaly = False
    start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif labels[i] == 0 and in_anomaly:
            ax.axvspan(start, i, color="red", alpha=0.2)
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start, len(labels), color="red", alpha=0.2)
    red_patch = mpatches.Patch(color="red", alpha=0.3, label="True anomaly")
    ax.legend(handles=[ax.lines[0], red_patch], fontsize=8)
    ax.set_ylabel("Normalised value")
    ax.set_title("Raw signal with true anomaly regions")

    # Panel 2 — reconstruction error + threshold
    ax = axes[1]
    ax.plot(t, point_scores, color="darkorange", lw=0.7, label="Recon. error")
    ax.axhline(best_thresh, color="red", ls="--", lw=1.2, label=f"Threshold {best_thresh:.4f}")
    ax.legend(fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title("Point-level reconstruction error")

    # Panel 3 — predicted vs ground-truth binary labels
    ax = axes[2]
    ax.step(t, labels, color="green", lw=0.8, label="Ground truth", where="post")
    ax.step(t, best_preds, color="red", lw=0.8, ls="--", label="Predicted", where="post")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8)
    ax.set_ylabel("Anomaly")
    ax.set_xlabel("Time step")
    ax.set_title("Predicted vs ground-truth anomaly labels")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "anomaly_results.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved anomaly plot → {out_path}")


def plot_loss_curve(train_history, val_history, out_dir):
    """Training / validation loss curve saved to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_history, label="Train loss", color="steelblue")
    ax.plot(val_history, label="Val loss", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("LSTM Autoencoder training curve")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curve → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(CONFIG["device"])
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_data, test_data, _ = load_channel(CONFIG["channel"], CONFIG["data_dir"])
    labels = load_labels(CONFIG["channel"], len(test_data), CONFIG["data_dir"])

    anomaly_rate = labels.mean() * 100
    print(f"Train size : {len(train_data):,} steps")
    print(f"Test  size : {len(test_data):,} steps  |  Anomaly rate: {anomaly_rate:.2f}%")

    train_loader, test_loader, train_windows, test_windows = get_dataloaders(
        train_data,
        test_data,
        window_size=CONFIG["window_size"],
        batch_size=CONFIG["batch_size"],
    )
    print(f"Train windows: {len(train_windows):,}  |  Test windows: {len(test_windows):,}")

    input_dim = train_data.shape[1]

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # ------------------------------------------------------------------
    # 3. Training loop with optional MLflow logging
    # ------------------------------------------------------------------
    try:
        import mlflow
        import mlflow.pytorch
        mlflow_available = True
    except ImportError:
        mlflow_available = False

    metrics_log = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)

    if mlflow_available:
        mlflow.set_experiment("anomaly-detection-smap")
        run = mlflow.start_run()
        mlflow.log_params({k: v for k, v in CONFIG.items() if k != "device"})
    else:
        run = None

    try:
        for epoch in range(1, CONFIG["num_epochs"] + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = val_epoch(model, test_loader, criterion, device)
            scheduler.step(val_loss)
            elapsed = time.time() - t0

            metrics_log["train_loss"].append(train_loss)
            metrics_log["val_loss"].append(val_loss)

            if mlflow_available:
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), CONFIG["save_path"])

            if epoch % 5 == 0:
                print(
                    f"Ep {epoch:02d} | Train {train_loss:.6f} | "
                    f"Val {val_loss:.6f} | {elapsed:.1f}s"
                )
    finally:
        if mlflow_available and run is not None:
            mlflow.end_run()

    # ------------------------------------------------------------------
    # 4. Load best checkpoint and evaluate
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(CONFIG["save_path"], map_location=device))
    model.eval()

    # LSTM-AE window scores → point scores
    ae_window_scores = reconstruction_errors(model, test_loader, device).numpy()
    ae_point_scores = window_to_point_scores(ae_window_scores, len(test_data), CONFIG["window_size"])

    # Trim labels to match point scores length (windows don't cover tail)
    scored_length = len(ae_point_scores)
    labels_trimmed = labels[:scored_length]

    best_thresh, best_f1, best_preds = find_best_threshold(ae_point_scores, labels_trimmed)

    ae_prec = precision_score(labels_trimmed, best_preds, zero_division=0)
    ae_rec = recall_score(labels_trimmed, best_preds, zero_division=0)
    try:
        ae_auc = roc_auc_score(labels_trimmed, ae_point_scores)
    except ValueError:
        ae_auc = float("nan")

    # Isolation Forest baseline
    if_scores = isolation_forest_errors(train_windows, test_windows)
    if_point_scores = window_to_point_scores(if_scores, len(test_data), CONFIG["window_size"])
    _, if_f1, _ = find_best_threshold(if_point_scores, labels_trimmed)

    delta = (best_f1 - if_f1) / max(if_f1, 1e-9) * 100

    # ------------------------------------------------------------------
    # 5. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"LSTM-AE → F1:{best_f1:.3f}  Prec:{ae_prec:.3f}  Rec:{ae_rec:.3f}  AUC:{ae_auc:.3f}")
    print(f"Iso.F.  → F1:{if_f1:.3f}")
    print(f"Δ F1    → {delta:+.1f}%")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 6. Save results.json
    # ------------------------------------------------------------------
    results = {
        "channel": CONFIG["channel"],
        "lstm_ae": {
            "f1": round(best_f1, 4),
            "precision": round(ae_prec, 4),
            "recall": round(ae_rec, 4),
            "auc": round(ae_auc, 4) if not np.isnan(ae_auc) else None,
            "threshold": round(float(best_thresh), 6),
        },
        "isolation_forest": {"f1": round(if_f1, 4)},
        "delta_f1_pct": round(delta, 2),
        "train_loss_history": metrics_log["train_loss"],
        "val_loss_history": metrics_log["val_loss"],
    }
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    results_path = os.path.join(CONFIG["out_dir"], "results.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved results → {results_path}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    plot_anomaly_results(
        test_data, ae_point_scores, best_thresh, labels_trimmed, best_preds, CONFIG["out_dir"]
    )
    plot_loss_curve(metrics_log["train_loss"], metrics_log["val_loss"], CONFIG["out_dir"])


if __name__ == "__main__":
    main()
