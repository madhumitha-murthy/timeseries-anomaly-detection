"""
drift_monitor.py — Data drift detection for the LSTM Autoencoder inference API.

How it works
------------
During training, train.py computes reconstruction errors on the training set and
saves a representative sample to models/baseline_stats.json.  In production, the
API accumulates reconstruction errors from incoming inference requests in a
rolling window.  When the window is full, DriftMonitor runs a two-sample
Kolmogorov-Smirnov test between the baseline sample and the recent window.

A low KS p-value (< alpha) indicates that the incoming data distribution has
shifted away from the training distribution — a signal that the model may be
operating out-of-distribution and should be re-evaluated or retrained.

Usage (inside api.py)
---------------------
    monitor = DriftMonitor.from_file("../models/baseline_stats.json")
    monitor.update(reconstruction_error)
    status = monitor.check()
    # status.drift_detected, status.p_value, status.mean_shift_sigmas
"""

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftStatus:
    """Result of a single drift check."""
    drift_detected: bool
    p_value: Optional[float]          # KS test p-value; None if window not full yet
    ks_statistic: Optional[float]     # KS test statistic (effect size)
    mean_shift_sigmas: Optional[float]  # (recent_mean - baseline_mean) / baseline_std
    window_size: int                  # number of samples in the recent window
    baseline_size: int                # number of samples in the baseline
    message: str


class DriftMonitor:
    """
    Rolling-window drift detector based on the two-sample KS test.

    Parameters
    ----------
    baseline_sample : list[float]
        Reconstruction errors from the training set (saved by train.py).
    window_size : int
        Number of recent inference scores to accumulate before running a KS test.
        Default 200 — large enough for reliable statistics, small enough to detect
        drift within a few minutes of sustained traffic.
    alpha : float
        Significance level for the KS test.  Default 0.05 (5% false-positive rate).
    mean_shift_threshold : float
        Additional trigger: if the recent mean deviates by more than this many
        standard deviations from the baseline mean, flag as drift regardless of
        the KS p-value.  Default 3.0 sigmas.
    """

    def __init__(
        self,
        baseline_sample: list,
        window_size: int = 200,
        alpha: float = 0.05,
        mean_shift_threshold: float = 3.0,
    ):
        self.baseline = np.array(baseline_sample, dtype=np.float32)
        self.baseline_mean = float(np.mean(self.baseline))
        self.baseline_std = float(np.std(self.baseline)) or 1e-8  # guard divide-by-zero
        self.window_size = window_size
        self.alpha = alpha
        self.mean_shift_threshold = mean_shift_threshold
        self._buffer: deque = deque(maxlen=window_size)
        self._total_checks: int = 0
        self._drift_count: int = 0

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "DriftMonitor":
        """Load baseline statistics from the JSON file written by train.py."""
        if not os.path.exists(path):
            logger.warning(
                "Baseline stats file not found at %s. "
                "DriftMonitor will operate in no-baseline mode (drift checks disabled).",
                path,
            )
            return cls(baseline_sample=[], **kwargs)
        with open(path) as fh:
            data = json.load(fh)
        sample = data.get("sample", [])
        if not sample:
            logger.warning("baseline_stats.json has no 'sample' key — re-run train.py.")
            return cls(baseline_sample=[], **kwargs)
        logger.info(
            "DriftMonitor loaded baseline: %d samples, mean=%.4f, std=%.4f",
            len(sample),
            data.get("mean", 0.0),
            data.get("std", 0.0),
        )
        return cls(baseline_sample=sample, **kwargs)

    def update(self, reconstruction_error: float) -> None:
        """Add a single inference reconstruction error to the rolling window."""
        self._buffer.append(float(reconstruction_error))

    def check(self) -> DriftStatus:
        """
        Run a drift check against the baseline.

        Returns a DriftStatus immediately.  If the rolling window has fewer than
        window_size samples, or if no baseline is loaded, drift_detected=False and
        p_value=None (insufficient data — not a clean bill of health).
        """
        n_recent = len(self._buffer)
        n_baseline = len(self.baseline)

        if n_baseline == 0:
            return DriftStatus(
                drift_detected=False,
                p_value=None,
                ks_statistic=None,
                mean_shift_sigmas=None,
                window_size=n_recent,
                baseline_size=0,
                message="No baseline loaded — drift detection disabled.",
            )

        if n_recent < self.window_size:
            return DriftStatus(
                drift_detected=False,
                p_value=None,
                ks_statistic=None,
                mean_shift_sigmas=None,
                window_size=n_recent,
                baseline_size=n_baseline,
                message=f"Accumulating window ({n_recent}/{self.window_size} samples).",
            )

        recent = np.array(self._buffer, dtype=np.float32)
        ks_stat, p_value = stats.ks_2samp(self.baseline, recent)
        recent_mean = float(np.mean(recent))
        mean_shift_sigmas = (recent_mean - self.baseline_mean) / self.baseline_std

        ks_drift = p_value < self.alpha
        mean_drift = abs(mean_shift_sigmas) > self.mean_shift_threshold
        drift_detected = ks_drift or mean_drift

        self._total_checks += 1
        if drift_detected:
            self._drift_count += 1

        if drift_detected:
            msg = (
                f"DRIFT DETECTED — KS p={p_value:.4f} (alpha={self.alpha}), "
                f"mean shift={mean_shift_sigmas:+.2f}σ. "
                f"Consider retraining or investigating upstream data changes."
            )
            logger.warning(msg)
        else:
            msg = (
                f"No drift — KS p={p_value:.4f}, "
                f"mean shift={mean_shift_sigmas:+.2f}σ."
            )

        return DriftStatus(
            drift_detected=drift_detected,
            p_value=round(float(p_value), 6),
            ks_statistic=round(float(ks_stat), 6),
            mean_shift_sigmas=round(float(mean_shift_sigmas), 4),
            window_size=n_recent,
            baseline_size=n_baseline,
            message=msg,
        )

    @property
    def stats(self) -> dict:
        """Summary statistics for the /drift/status endpoint."""
        return {
            "buffer_fill": len(self._buffer),
            "window_size": self.window_size,
            "baseline_size": len(self.baseline),
            "baseline_mean": round(self.baseline_mean, 6),
            "baseline_std": round(self.baseline_std, 6),
            "total_checks": self._total_checks,
            "drift_events": self._drift_count,
        }
