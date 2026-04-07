"""
lp_optimizer.py — LP-based anomaly segment triage under an inspection budget

Business problem
----------------
The LSTM Autoencoder scores every test time-step.  Thresholding those scores
yields *candidate anomaly segments* — contiguous runs of flagged points.  In a
real deployment a maintenance team must physically inspect flagged regions, and
inspection is expensive: each time-step costs operator time.

Given a fixed inspection budget (e.g. 10 % of the test period), the team needs
to decide *which* segments to prioritise.  The obvious heuristic — sort by
anomaly score, pick the top ones — is suboptimal because it ignores inspection
cost.  A segment with the highest raw score might be so long that it consumes
the entire budget, preventing the team from inspecting several shorter but
high-density segments that together contain more anomaly signal.

The correct formulation is a Linear Programme.

LP formulation — Fractional Knapsack
─────────────────────────────────────
    Input:
      S candidate segments, each with
        score_s  : mean LSTM-AE reconstruction error  (anomaly signal density)
        length_s : number of time-steps               (inspection cost / weight)
      B : budget_steps  (maximum time-steps the team can inspect)

    Decision variables:
      x_s ∈ [0, 1]   fraction of segment s to inspect
        x_s = 1 → fully prioritise  |  x_s = 0 → skip

    Objective  (maximise anomaly signal covered):
      maximise   Σ_s  score_s · x_s

    Constraints:
      Σ_s  length_s · x_s  ≤  B        (budget constraint)
      0  ≤  x_s  ≤  1   for all s       (variable bounds)

    scipy.optimize.linprog minimises, so we negate: minimise  −score^T x.

Why LP beats naive greedy
──────────────────────────
Naive greedy sorts by raw score and picks the top segments until the budget is
exhausted.  This is suboptimal when a segment has a high absolute score but a
poor score/length ratio (low anomaly density).

Example (budget = 5 steps):
  Segment A : score = 3.0 · length = 10  →  density = 0.30
  Segment B : score = 2.5 · length =  2  →  density = 1.25

  Naive greedy picks A first (highest score):
    fills 5 of 10 steps → objective = 3.0 × 0.5 = 1.50

  LP (optimal) picks B fully then A partially:
    B: 2 steps, contribution = 2.50
    A: 3 steps, contribution = 3.0 × 0.3 = 0.90
    total objective = 3.40   (+127 % vs naive greedy)

For the fractional knapsack the LP optimum equals the greedy optimum only when
greedy sorts by score/length density — NOT by raw score.  By running linprog
explicitly we get the provably optimal solution and keep the door open for
additional operational constraints (per-channel caps, inspection-gap rules,
safety-floor requirements) without rewriting the solver.

Expected latency : < 1 ms for S ≤ 1 000 segments (HiGHS back-end, CPU).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def extract_anomaly_candidates(
    point_scores: np.ndarray,
    threshold: float,
) -> list[dict]:
    """Find contiguous runs of above-threshold points as candidate segments.

    Parameters
    ----------
    point_scores : np.ndarray, shape (T,)
        Point-level anomaly scores (output of window_to_point_scores).
    threshold : float
        Score value above which a point is considered anomalous.

    Returns
    -------
    segments : list of dicts, each with keys
        start  — inclusive start index
        end    — exclusive end index
        length — number of time-steps in the segment
        score  — mean reconstruction error (anomaly signal density)
    """
    segments: list[dict] = []
    in_seg = False
    seg_start = 0

    for i, s in enumerate(point_scores):
        if s >= threshold and not in_seg:
            seg_start = i
            in_seg = True
        elif s < threshold and in_seg:
            segments.append({
                "start":  seg_start,
                "end":    i,
                "length": i - seg_start,
                "score":  float(point_scores[seg_start:i].mean()),
            })
            in_seg = False

    if in_seg:
        segments.append({
            "start":  seg_start,
            "end":    len(point_scores),
            "length": len(point_scores) - seg_start,
            "score":  float(point_scores[seg_start:].mean()),
        })

    return segments


# ---------------------------------------------------------------------------
# Internal metric helper
# ---------------------------------------------------------------------------

def _compute_triage_metrics(
    segments: list[dict],
    x: np.ndarray,
    budget_steps: int,
) -> dict:
    """Compute objective, budget usage, and coverage for one triage solution.

    Parameters
    ----------
    segments     : list of segment dicts
    x            : allocation fractions, shape (S,)
    budget_steps : int  authorised inspection budget

    Returns
    -------
    dict with keys:
        objective              — Σ score_s · x_s  (anomaly signal captured)
        budget_used            — Σ length_s · x_s  (steps consumed)
        budget_utilization_pct — budget_used / budget_steps × 100
        coverage_pct           — objective / total_score × 100
        n_selected             — segments with x_s > 0.5
        top_segments           — selected segments sorted by priority desc
    """
    if not segments:
        return {
            "objective":              0.0,
            "budget_used":            0.0,
            "budget_utilization_pct": 0.0,
            "coverage_pct":           0.0,
            "n_selected":             0,
            "top_segments":           [],
        }

    total_score = float(sum(seg["score"] for seg in segments))
    objective   = float(sum(seg["score"]  * xi for seg, xi in zip(segments, x)))
    budget_used = float(sum(seg["length"] * xi for seg, xi in zip(segments, x)))

    budget_util = (100.0 * budget_used / budget_steps) if budget_steps > 0 else 0.0
    coverage    = (100.0 * objective   / total_score)  if total_score  > 0 else 0.0

    selected = sorted(
        [(seg, float(xi)) for seg, xi in zip(segments, x) if xi > 0.5],
        key=lambda t: -t[1],
    )

    return {
        "objective":              round(objective,   4),
        "budget_used":            round(budget_used, 1),
        "budget_utilization_pct": round(budget_util, 1),
        "coverage_pct":           round(coverage,    1),
        "n_selected":             len(selected),
        "top_segments": [
            {
                "start":    seg["start"],
                "end":      seg["end"],
                "score":    round(seg["score"], 6),
                "priority": round(xi, 4),
            }
            for seg, xi in selected
        ],
    }


# ---------------------------------------------------------------------------
# LP solver
# ---------------------------------------------------------------------------

def lp_triage(
    point_scores: np.ndarray,
    threshold: float,
    budget_fraction: float = 0.10,
) -> tuple[list[dict], np.ndarray]:
    """LP-optimal anomaly segment selection under an inspection budget.

    Solves the fractional knapsack LP via scipy.optimize.linprog (HiGHS):
      minimise  −score^T x
      s.t.      length^T x  ≤  budget_steps
                0 ≤ x_s ≤ 1

    Parameters
    ----------
    point_scores    : np.ndarray, shape (T,)
        Real LSTM-AE point-level reconstruction errors from window_to_point_scores.
    threshold       : float
        Deployment threshold (99th pct of training errors).
    budget_fraction : float, default 0.10
        Fraction of T available for inspection.

    Returns
    -------
    segments : list of S dicts (start, end, length, score)
    x        : np.ndarray, shape (S,), LP-optimal fractions in [0, 1]
    """
    budget_steps = max(1, int(budget_fraction * len(point_scores)))
    segments = extract_anomaly_candidates(point_scores, threshold)

    if not segments:
        return [], np.array([], dtype=np.float64)

    n       = len(segments)
    scores  = np.array([seg["score"]  for seg in segments], dtype=np.float64)
    lengths = np.array([seg["length"] for seg in segments], dtype=np.float64)

    # ── scipy.optimize.linprog — HiGHS back-end ─────────────────────────
    #
    #   minimise   c^T x
    #   s.t.       A_ub x  ≤  b_ub      (one budget-constraint row)
    #              0 ≤ x_s ≤ 1          (bounds per variable)
    #
    c    = -scores                          # maximise score ↔ minimise −score
    A_ub = lengths.reshape(1, -1)           # (1, S)
    b_ub = np.array([float(budget_steps)])

    result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(0.0, 1.0)] * n, method="highs")

    if result.success:
        return segments, result.x

    # Greedy fallback on solver failure (optimal for fractional knapsack)
    ratio     = scores / np.maximum(lengths, 1.0)
    order     = np.argsort(-ratio)
    x         = np.zeros(n, dtype=np.float64)
    remaining = float(budget_steps)
    for idx in order:
        if remaining <= 0.0:
            break
        frac      = min(1.0, remaining / lengths[idx])
        x[idx]    = frac
        remaining -= frac * lengths[idx]

    return segments, x


# ---------------------------------------------------------------------------
# Naive greedy baseline
# ---------------------------------------------------------------------------

def naive_greedy_triage(segments: list[dict], budget_steps: int) -> np.ndarray:
    """Naive greedy: sort by raw score, fill budget until exhausted.

    This is the intuitive but suboptimal heuristic.  Sorting by raw score
    ignores inspection cost (length), so a long segment with a high absolute
    score can consume the entire budget while shorter, denser segments that
    would yield more anomaly signal per step go uninspected.

    lp_triage solves the same problem exactly (provably optimal) by letting
    linprog implicitly rank by score/length density via the LP simplex.

    Parameters
    ----------
    segments     : list of dicts from extract_anomaly_candidates
    budget_steps : int

    Returns
    -------
    x : np.ndarray, shape (S,), greedy allocation fractions in [0, 1]
    """
    if not segments:
        return np.array([], dtype=np.float64)

    n       = len(segments)
    scores  = np.array([seg["score"]  for seg in segments], dtype=np.float64)
    lengths = np.array([seg["length"] for seg in segments], dtype=np.float64)

    # Sort by raw score descending — naive, ignores cost
    order     = np.argsort(-scores)
    x         = np.zeros(n, dtype=np.float64)
    remaining = float(budget_steps)

    for idx in order:
        if remaining <= 0.0:
            break
        frac      = min(1.0, remaining / lengths[idx])
        x[idx]    = frac
        remaining -= frac * lengths[idx]

    return x


# ---------------------------------------------------------------------------
# LP vs greedy comparison  (main entry point used by train.py)
# ---------------------------------------------------------------------------

def compare_lp_vs_greedy(
    point_scores: np.ndarray,
    threshold: float,
    budget_fraction: float = 0.10,
) -> dict:
    """Run LP and naive greedy on real LSTM-AE scores; return full comparison.

    Takes the actual point-level reconstruction errors from the trained model
    as input — not synthetic scores.

    Parameters
    ----------
    point_scores    : np.ndarray, shape (T,)
        Real LSTM-AE reconstruction errors from window_to_point_scores.
    threshold       : float
        Deployment threshold (99th pct of training errors).
    budget_fraction : float, default 0.10

    Returns
    -------
    dict with keys:
        n_candidates     — number of candidate segments found
        total_score      — Σ score_s (unconstrained maximum possible objective)
        budget_steps     — int  (budget in time-steps)
        budget_fraction  — float
        lp               — full metric dict for LP solution
        greedy           — full metric dict for naive greedy solution
        lp_gain_pct      — (lp_obj − greedy_obj) / greedy_obj × 100
                           0 when both are equal (they agree when all segments
                           fit in budget); positive when LP is strictly better
        lp_is_optimal    — always True: LP solves fractional knapsack exactly
    """
    budget_steps = max(1, int(budget_fraction * len(point_scores)))
    segments     = extract_anomaly_candidates(point_scores, threshold)

    total_score = float(sum(seg["score"] for seg in segments)) if segments else 0.0

    _, x_lp      = lp_triage(point_scores, threshold, budget_fraction)
    x_greedy     = naive_greedy_triage(segments, budget_steps)

    lp_metrics     = _compute_triage_metrics(segments, x_lp,     budget_steps)
    greedy_metrics = _compute_triage_metrics(segments, x_greedy, budget_steps)

    greedy_obj = greedy_metrics["objective"]
    lp_obj     = lp_metrics["objective"]
    lp_gain    = (
        round(100.0 * (lp_obj - greedy_obj) / greedy_obj, 2)
        if greedy_obj > 0 else 0.0
    )

    return {
        "n_candidates":    len(segments),
        "total_score":     round(total_score, 4),
        "budget_steps":    budget_steps,
        "budget_fraction": budget_fraction,
        "lp":              lp_metrics,
        "greedy":          greedy_metrics,
        "lp_gain_pct":     lp_gain,
        "lp_is_optimal":   True,
    }


# ---------------------------------------------------------------------------
# Legacy summary helper (kept for backward compatibility / existing tests)
# ---------------------------------------------------------------------------

def lp_triage_summary(segments: list[dict], x: np.ndarray) -> dict:
    """Human-readable summary of an LP triage solution.

    Kept for backward compatibility — prefer compare_lp_vs_greedy for new code.

    Returns
    -------
    dict with keys: n_candidates, n_selected, steps_inspected, top_segments
    """
    if not segments:
        return {
            "n_candidates":    0,
            "n_selected":      0,
            "steps_inspected": 0.0,
            "top_segments":    [],
        }

    steps_inspected = float(sum(seg["length"] * xi for seg, xi in zip(segments, x)))

    selected = sorted(
        [(seg, float(xi)) for seg, xi in zip(segments, x) if xi > 0.5],
        key=lambda t: -t[1],
    )

    return {
        "n_candidates":    len(segments),
        "n_selected":      len(selected),
        "steps_inspected": round(steps_inspected, 1),
        "top_segments": [
            {
                "start":    seg["start"],
                "end":      seg["end"],
                "score":    round(seg["score"], 6),
                "priority": round(xi, 4),
            }
            for seg, xi in selected
        ],
    }
