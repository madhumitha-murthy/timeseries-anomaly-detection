"""
train.py — Full training pipeline for LSTM Autoencoder on NASA SMAP dataset.

Usage:
    cd src && python train.py
    cd src && python train.py --channel P-1 --num_epochs 50
    cd src && python train.py --config ../configs/experiment_2.yaml

Expects the SMAP data under data_dir with sub-folders train/ and test/
and the file labeled_anomalies.csv.
"""

import argparse
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
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset import get_dataloaders, load_channel, load_labels, train_val_split
from lp_optimizer import (
    compare_lp_vs_greedy,
    extract_anomaly_candidates,
    lp_triage,
    naive_greedy_triage,
)
from des_simulator import compare_des_schedules
from model import LSTMAutoencoder, isolation_forest_errors, reconstruction_errors

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "channel":      "E-7",
    "data_dir":     "../data",
    "hidden_dim":   64,
    "num_layers":   1,
    "dropout":      0.0,
    "window_size":  30,
    "batch_size":   32,
    "num_epochs":   200,
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "save_path":    "../models/lstm_ae_best.pth",
    "out_dir":      "../outputs/",
}


def load_config(config_path: str = None, **overrides) -> dict:
    """Load configuration from a YAML file and apply CLI overrides.

    Priority (highest → lowest): CLI overrides > YAML file > built-in defaults.

    Parameters
    ----------
    config_path : str or None
        Path to a YAML config file.  If None, only defaults + overrides are used.
    **overrides
        Key/value pairs from argparse (only non-None values override the file).

    Returns
    -------
    cfg : dict  Complete configuration ready for use in main().
    """
    cfg = dict(_DEFAULTS)

    if config_path is not None:
        with open(config_path) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg.update(file_cfg)

    # Apply explicit CLI overrides (skip None so argparse defaults don't shadow file values)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    # device is always auto-detected — never read from file or CLI
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train LSTM Autoencoder for anomaly detection on NASA SMAP data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       type=str,   default=None,  help="Path to YAML config file (overrides built-in defaults)")
    p.add_argument("--channel",      type=str,   default=None,  help="SMAP channel ID, e.g. E-7")
    p.add_argument("--data_dir",     type=str,   default=None,  help="Root data directory")
    p.add_argument("--hidden_dim",   type=int,   default=None,  help="LSTM hidden units")
    p.add_argument("--num_layers",   type=int,   default=None,  help="Number of LSTM layers")
    p.add_argument("--dropout",      type=float, default=None,  help="LSTM dropout probability")
    p.add_argument("--window_size",  type=int,   default=None,  help="Sliding window length")
    p.add_argument("--batch_size",   type=int,   default=None,  help="Mini-batch size")
    p.add_argument("--num_epochs",   type=int,   default=None,  help="Maximum training epochs")
    p.add_argument("--lr",           type=float, default=None,  help="Adam learning rate")
    p.add_argument("--weight_decay", type=float, default=None,  help="Adam weight decay")
    p.add_argument("--save_path",    type=str,   default=None,  help="Path to save best model checkpoint")
    p.add_argument("--out_dir",      type=str,   default=None,  help="Directory for plots and results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when validation loss stops improving.

    Prevents the ~70 wasted epochs seen in Run 4 where val loss was flat from
    epoch 130 onward. patience=15 gives the scheduler room to reduce LR and
    recover before stopping.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> dict:
    """Compute the full evaluation metric suite for one model/threshold combo.

    Parameters
    ----------
    scores    : continuous anomaly scores, shape (T,)
    labels    : binary ground-truth, shape (T,)
    preds     : binary predictions at threshold, shape (T,)
    threshold : the threshold used to produce preds

    Returns
    -------
    dict with keys: f1, precision, recall, auc_roc, avg_precision,
                    false_alarm_rate, detection_delay_steps, threshold
    """
    f1   = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)

    try:
        auc_roc = roc_auc_score(labels, scores)
    except ValueError:
        auc_roc = float("nan")

    try:
        avg_prec = average_precision_score(labels, scores)
    except ValueError:
        avg_prec = float("nan")

    # False Alarm Rate — FP / (FP + TN) — fraction of normal steps wrongly flagged.
    # Critical in manufacturing: every false alarm is a potential unplanned downtime.
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Detection Delay — median number of steps between the START of each anomaly
    # segment and the first prediction of 1 within that segment.
    # Lower = earlier warning = more time to act before failure.
    detection_delays = []
    in_anomaly = False
    seg_start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            seg_start = i
            in_anomaly = True
        elif (labels[i] == 0 or i == len(labels) - 1) and in_anomaly:
            seg_end = i if labels[i] == 0 else i + 1
            seg_preds = preds[seg_start:seg_end]
            flagged = np.where(seg_preds == 1)[0]
            if len(flagged) > 0:
                detection_delays.append(int(flagged[0]))  # steps into segment
            else:
                detection_delays.append(seg_end - seg_start)  # missed = full length
            in_anomaly = False

    detection_delay = float(np.median(detection_delays)) if detection_delays else None

    return {
        "threshold":             round(float(threshold), 6),
        "f1":                    round(float(f1), 4),
        "precision":             round(float(prec), 4),
        "recall":                round(float(rec), 4),
        "auc_roc":               round(float(auc_roc), 4) if auc_roc is not None and not np.isnan(auc_roc) else None,
        "avg_precision":         round(float(avg_prec), 4) if avg_prec is not None and not np.isnan(avg_prec) else None,
        "false_alarm_rate":      round(float(false_alarm_rate), 4),
        "detection_delay_steps": round(detection_delay, 1) if detection_delay is not None else None,
    }


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
# LP comparison printer
# ---------------------------------------------------------------------------

def _print_lp_comparison(cmp: dict, total_steps: int) -> None:
    """Print the LP vs naive-greedy comparison table to stdout."""
    W = 72
    lp  = cmp["lp"]
    gr  = cmp["greedy"]

    print("\n" + "=" * W)
    print("LP TRIAGE — Anomaly Segment Inspection (LSTM-AE reconstruction errors)")
    print(
        f"  Budget : {cmp['budget_fraction']*100:.0f}% of {total_steps} test steps"
        f" = {cmp['budget_steps']} steps"
    )
    print(f"  Candidate segments : {cmp['n_candidates']}")
    print(f"  Total anomaly score available : {cmp['total_score']:.4f}")
    print()
    print(f"  {'':22s}  {'Objective':>10}  {'Budget used':>11}  "
          f"{'Utilization':>11}  {'Coverage':>9}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*9}")
    print(f"  {'LP  (scipy linprog)':22s}  {lp['objective']:>10.4f}  "
          f"{lp['budget_used']:>9.1f} s  "
          f"{lp['budget_utilization_pct']:>9.1f} %  "
          f"{lp['coverage_pct']:>7.1f} %")
    print(f"  {'Greedy (naive)':22s}  {gr['objective']:>10.4f}  "
          f"{gr['budget_used']:>9.1f} s  "
          f"{gr['budget_utilization_pct']:>9.1f} %  "
          f"{gr['coverage_pct']:>7.1f} %")
    print()

    if cmp["lp_gain_pct"] > 0:
        print(
            f"  LP gain : +{cmp['lp_gain_pct']:.1f} % more anomaly signal vs naive greedy"
        )
    else:
        print("  LP gain : 0 % — both methods agree (all segments fit in budget)")
    print("  LP is provably optimal for fractional knapsack.")

    if lp["top_segments"]:
        print()
        print("  Segments selected by LP  (priority > 0.5, sorted desc):")
        for i, seg in enumerate(lp["top_segments"][:5], 1):
            print(
                f"    {i}.  steps [{seg['start']:5d} – {seg['end']:5d}]"
                f"  score={seg['score']:.4f}  priority={seg['priority']:.3f}"
            )
        if lp["n_selected"] > 5:
            print(f"    … and {lp['n_selected'] - 5} more segment(s)")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: dict = None):
    """Run the full training and evaluation pipeline.

    Parameters
    ----------
    config : dict or None
        Pre-built configuration dict (e.g., from a notebook).  When None,
        configuration is loaded from CLI arguments and/or a YAML file.
    """
    if config is None:
        args = parse_args()
        cfg = load_config(
            config_path=args.config,
            channel=args.channel,
            data_dir=args.data_dir,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_path=args.save_path,
            out_dir=args.out_dir,
        )
    else:
        cfg = dict(_DEFAULTS)
        cfg.update(config)
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    train_data, test_data, _ = load_channel(cfg["channel"], cfg["data_dir"])
    labels = load_labels(cfg["channel"], len(test_data), cfg["data_dir"])

    anomaly_rate = labels.mean()
    print(f"Train size : {len(train_data):,} steps")
    print(f"Test  size : {len(test_data):,} steps  |  Anomaly rate: {anomaly_rate * 100:.2f}%")

    # Split train into train/val (80/20, time-ordered).
    # Val comes from the same anomaly-free training distribution — no leakage.
    # Previously val_epoch used the test set, meaning the saved checkpoint was
    # selected using test data — a form of leakage.
    train_split, val_split = train_val_split(train_data, val_frac=0.2)

    train_loader, val_loader, train_windows, _ = get_dataloaders(
        train_split,
        val_split,
        window_size=cfg["window_size"],
        batch_size=cfg["batch_size"],
    )

    # Separate test loader — never touched during training or checkpoint selection.
    _, test_loader, _, test_windows = get_dataloaders(
        train_split,
        test_data,
        window_size=cfg["window_size"],
        batch_size=cfg["batch_size"],
    )

    print(
        f"Train windows: {len(train_windows):,}  |  "
        f"Val windows: {len(val_loader.dataset):,}  |  "
        f"Test windows: {len(test_windows):,}"
    )

    input_dim = train_data.shape[1]

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
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
    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)

    early_stopper = EarlyStopping(patience=15, min_delta=1e-5)

    if mlflow_available:
        mlflow.set_experiment("anomaly-detection-smap")
        run = mlflow.start_run()
        mlflow.log_params({k: v for k, v in cfg.items() if k != "device"})
    else:
        run = None

    try:
        for epoch in range(1, cfg["num_epochs"] + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            # val_loader is built from train_split[:val] — NOT the test set.
            val_loss = val_epoch(model, val_loader, criterion, device)
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
                torch.save(model.state_dict(), cfg["save_path"])

            if epoch % 5 == 0:
                print(
                    f"Ep {epoch:03d} | Train {train_loss:.6f} | "
                    f"Val {val_loss:.6f} | {elapsed:.1f}s"
                )

            if early_stopper.step(val_loss):
                print(f"Early stopping at epoch {epoch} (no val improvement for {early_stopper.patience} epochs)")
                break
    finally:
        if mlflow_available and run is not None:
            mlflow.end_run()

    # ------------------------------------------------------------------
    # 4. Load best checkpoint and evaluate
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(cfg["save_path"], map_location=device))
    model.eval()

    # LSTM-AE window scores → point scores
    ae_window_scores = reconstruction_errors(model, test_loader, device).numpy()
    ae_point_scores = window_to_point_scores(ae_window_scores, len(test_data), cfg["window_size"])

    # Trim labels to match point scores length (windows don't cover tail)
    scored_length = len(ae_point_scores)
    labels_trimmed = labels[:scored_length]

    # ------------------------------------------------------------------
    # Deployment threshold — derived from TRAINING error distribution only.
    #
    # Previously find_best_threshold() swept 200 values against the labelled
    # test set and reported the best F1.  That is an oracle: in production
    # you never have test labels to select a threshold.  The correct approach
    # is to set the threshold from the training reconstruction errors (e.g.,
    # 99th percentile), then report what F1 that threshold achieves on test.
    # ------------------------------------------------------------------
    train_errors = reconstruction_errors(model, train_loader, device).numpy()
    deployment_threshold = float(np.percentile(train_errors, 99))

    # Save baseline reconstruction error statistics for drift monitoring.
    # The drift monitor (api.py) compares incoming window scores against this
    # training distribution using a KS test to detect data/concept drift.
    baseline_stats_path = os.path.join(os.path.dirname(cfg["save_path"]), "baseline_stats.json")
    with open(baseline_stats_path, "w") as fh:
        json.dump(
            {
                "mean": float(np.mean(train_errors)),
                "std": float(np.std(train_errors)),
                "p50": float(np.percentile(train_errors, 50)),
                "p95": float(np.percentile(train_errors, 95)),
                "p99": float(np.percentile(train_errors, 99)),
                # Store up to 1 000 samples — sufficient for a reliable KS test
                # while keeping the file small (~50 KB).
                "sample": train_errors[:1000].tolist(),
            },
            fh,
            indent=2,
        )
    print(f"Baseline stats saved → {baseline_stats_path}")

    deploy_preds = (ae_point_scores >= deployment_threshold).astype(int)

    # Save deployment threshold so the API can load the correct value
    # instead of relying on a hardcoded env-var default.
    threshold_path = os.path.join(os.path.dirname(cfg["save_path"]), "threshold.json")
    with open(threshold_path, "w") as fh:
        json.dump({"threshold": deployment_threshold}, fh, indent=2)
    print(f"Deployment threshold (99th pct of train errors): {deployment_threshold:.6f}")
    print(f"Saved threshold → {threshold_path}")

    deploy_metrics = compute_metrics(ae_point_scores, labels_trimmed, deploy_preds, deployment_threshold)

    # Oracle threshold (for reference only — NOT used by the deployed API)
    oracle_thresh, _, oracle_preds = find_best_threshold(ae_point_scores, labels_trimmed)
    oracle_metrics = compute_metrics(ae_point_scores, labels_trimmed, oracle_preds, oracle_thresh)

    # Isolation Forest baseline — calibrated to the actual anomaly rate.
    # Previously used contamination="auto" (≈10%) on a channel with 3.4%
    # anomaly rate, causing IsolationForest to over-flag and score unfairly low.
    if_contamination = max(float(anomaly_rate), 0.01)
    if_scores = isolation_forest_errors(train_windows, test_windows, contamination=if_contamination)
    if_point_scores = window_to_point_scores(if_scores, len(test_data), cfg["window_size"])
    _, _, if_preds = find_best_threshold(if_point_scores, labels_trimmed)
    if_thresh, _, _ = find_best_threshold(if_point_scores, labels_trimmed)
    if_metrics = compute_metrics(if_point_scores, labels_trimmed, if_preds, if_thresh)

    # ------------------------------------------------------------------
    # 5. LP triage — select which predicted anomaly segments to inspect
    #    given a 10 % inspection budget (fraction of the test period).
    #    Uses real LSTM-AE reconstruction errors (ae_point_scores).
    #    Compares LP (provably optimal) vs naive greedy (sort by score).
    # ------------------------------------------------------------------
    lp_comparison = compare_lp_vs_greedy(
        ae_point_scores, deployment_threshold, budget_fraction=0.10
    )
    _print_lp_comparison(lp_comparison, scored_length)

    # ------------------------------------------------------------------
    # 5b. DES simulation — quantify operational impact of LP vs greedy
    #     scheduling on the downstream inspection workflow.
    #     Calls lp_triage and naive_greedy_triage separately to obtain the
    #     raw allocation arrays needed by compare_des_schedules.
    # ------------------------------------------------------------------
    budget_steps_des   = max(1, int(0.10 * scored_length))
    segments_for_des   = extract_anomaly_candidates(ae_point_scores, deployment_threshold)
    _, x_lp_arr        = lp_triage(ae_point_scores, deployment_threshold, 0.10)
    x_greedy_arr       = naive_greedy_triage(segments_for_des, budget_steps_des)

    des_cmp = compare_des_schedules(
        segments_for_des, x_lp_arr, x_greedy_arr,
        n_machines=2, mttf=50.0, mttr=5.0, seed=42,
    )

    print("\n" + "=" * 80)
    print("DES SIMULATION — LP vs Greedy Inspection Schedule")
    print(f"  Segments: {des_cmp['lp'].n_jobs} LP jobs | {des_cmp['greedy'].n_jobs} greedy jobs")
    print(f"  Machines: {des_cmp['n_machines']}  |  Breakdowns: {'on (MTTF=50, MTTR=5)' if des_cmp['breakdown_enabled'] else 'off'}")
    print(f"  {'Metric':<28} {'LP':>12} {'Greedy':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Makespan':<28} {des_cmp['lp'].makespan:>12.2f} {des_cmp['greedy'].makespan:>12.2f}")
    print(f"  {'Mean wait time':<28} {des_cmp['lp'].mean_wait_time:>12.4f} {des_cmp['greedy'].mean_wait_time:>12.4f}")
    print(f"  {'P95 wait time':<28} {des_cmp['lp'].p95_wait_time:>12.4f} {des_cmp['greedy'].p95_wait_time:>12.4f}")
    print(f"  {'Machine utilisation':<28} {des_cmp['lp'].machine_utilisation:>12.4f} {des_cmp['greedy'].machine_utilisation:>12.4f}")
    print(f"  {'Throughput (jobs/t)':<28} {des_cmp['lp'].throughput:>12.4f} {des_cmp['greedy'].throughput:>12.4f}")
    print(f"  {'Breakdowns':<28} {des_cmp['lp'].breakdown_count:>12} {des_cmp['greedy'].breakdown_count:>12}")
    print(f"  LP wait reduction  : {des_cmp['lp_wait_reduction_pct']:+.2f} %")
    print(f"  LP makespan reduction: {des_cmp['lp_makespan_reduction_pct']:+.2f} %")
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 6. Print comparison table
    # ------------------------------------------------------------------
    def _row(name, m):
        return (
            f"  F1:{m['f1']:.3f}  Prec:{m['precision']:.3f}  Rec:{m['recall']:.3f}  "
            f"AUC:{m['auc_roc'] or 'n/a'}  AP:{m['avg_precision'] or 'n/a'}  "
            f"FAR:{m['false_alarm_rate']:.3f}  Delay:{m['detection_delay_steps']:.0f}steps"
        )

    print("\n" + "=" * 80)
    print("LSTM-AE (deployment threshold — production-realistic)")
    print(_row("deploy", deploy_metrics))
    print("LSTM-AE (oracle threshold — evaluation ceiling, NOT for deployment)")
    print(_row("oracle", oracle_metrics))
    print(f"Iso.Forest (contamination={if_contamination:.3f})")
    print(_row("if", if_metrics))
    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # 7. Save results.json
    # ------------------------------------------------------------------
    results = {
        "channel": cfg["channel"],
        "lstm_ae_deployment": {
            "threshold_source": "99th percentile of training reconstruction errors",
            **deploy_metrics,
        },
        "lstm_ae_oracle": {
            "threshold_source": "oracle sweep over test labels — ceiling only, not for deployment",
            **oracle_metrics,
        },
        "isolation_forest": {
            "contamination_used": round(if_contamination, 4),
            **if_metrics,
        },
        "train_loss_history": metrics_log["train_loss"],
        "val_loss_history":   metrics_log["val_loss"],
        "lp_triage":          lp_comparison,
        "des_simulation": {
            "n_machines":                des_cmp["n_machines"],
            "breakdown_enabled":         des_cmp["breakdown_enabled"],
            "lp_wait_reduction_pct":     des_cmp["lp_wait_reduction_pct"],
            "lp_makespan_reduction_pct": des_cmp["lp_makespan_reduction_pct"],
            "lp": {
                "schedule_name":        des_cmp["lp"].schedule_name,
                "n_jobs":               des_cmp["lp"].n_jobs,
                "makespan":             des_cmp["lp"].makespan,
                "mean_wait_time":       des_cmp["lp"].mean_wait_time,
                "p95_wait_time":        des_cmp["lp"].p95_wait_time,
                "mean_inspection_time": des_cmp["lp"].mean_inspection_time,
                "machine_utilisation":  des_cmp["lp"].machine_utilisation,
                "throughput":           des_cmp["lp"].throughput,
                "jobs_completed":       des_cmp["lp"].jobs_completed,
                "breakdown_count":      des_cmp["lp"].breakdown_count,
            },
            "greedy": {
                "schedule_name":        des_cmp["greedy"].schedule_name,
                "n_jobs":               des_cmp["greedy"].n_jobs,
                "makespan":             des_cmp["greedy"].makespan,
                "mean_wait_time":       des_cmp["greedy"].mean_wait_time,
                "p95_wait_time":        des_cmp["greedy"].p95_wait_time,
                "mean_inspection_time": des_cmp["greedy"].mean_inspection_time,
                "machine_utilisation":  des_cmp["greedy"].machine_utilisation,
                "throughput":           des_cmp["greedy"].throughput,
                "jobs_completed":       des_cmp["greedy"].jobs_completed,
                "breakdown_count":      des_cmp["greedy"].breakdown_count,
            },
        },
    }
    os.makedirs(cfg["out_dir"], exist_ok=True)
    results_path = os.path.join(cfg["out_dir"], "results.json")
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved results → {results_path}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    plot_anomaly_results(
        test_data, ae_point_scores, deployment_threshold, labels_trimmed, deploy_preds, cfg["out_dir"]
    )
    plot_loss_curve(metrics_log["train_loss"], metrics_log["val_loss"], cfg["out_dir"])

    return cfg, model, device


if __name__ == "__main__":
    main()
