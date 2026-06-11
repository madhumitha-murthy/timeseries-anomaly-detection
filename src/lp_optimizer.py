"""
lp_optimizer.py — LP-based anomaly segment triage under an inspection budget

Business problem
----------------
The LSTM Autoencoder scores every test time-step.  Thresholding those scores
yields *candidate anomaly segments* — contiguous runs of flagged points.  In a
real deployment a maintenance team must physically inspect flagged regions, and
inspection is expensive: each time-step costs operator time.

Given a fixed inspection budget (e.g. 10 % of the test period), the team needs
to decide *which* segments to prioritise.

LP formulation — Constrained Fractional Knapsack
─────────────────────────────────────────────────
    Decision variables:
      x_s ∈ [0, 1]   fraction of segment s to inspect

    Objective  (maximise anomaly signal covered):
      maximise   Σ_s  score_s · x_s

    Constraints:
      (C0)  Σ_s  length_s · x_s  ≤  B              total budget
      (C1)  length_s · x_s       ≤  cap · B   ∀ s  per-segment budget cap
            (no single segment may consume > cap fraction of budget;
             implemented as A_ub rows)
      (C2)  x_s  ≥  floor                    ∀ s in top-K by score
            (high-severity segments must receive minimum coverage;
             implemented as variable lower bounds)
      (C3)  0  ≤  x_s  ≤  1                 ∀ s  variable bounds

    scipy.optimize.linprog minimises, so we negate: minimise  −score^T x.

Why this is non-trivially different from density-greedy
───────────────────────────────────────────────────────
The unconstrained fractional knapsack is solvable in O(n log n) by sorting
segments by score/length density and filling greedily.  Adding C1 and C2
breaks that equivalence:

  • C1 (per-segment cap):  A high-density segment whose length alone would
    exhaust the cap is split.  The remaining budget flows to the next-best
    segment — density-greedy would have spent it all on the first.

  • C2 (min-coverage floor):  A high-priority segment with low density (long,
    moderate score) might be skipped entirely by density-greedy.  The floor
    forces a minimum inspection fraction, which may require the LP to
    re-balance the remaining budget across lower-density segments — a
    trade-off that density-greedy cannot express.

Honest comparison
─────────────────
  LP ≈ density_greedy   when C1 and C2 are NOT binding (small segments, loose
                        budget, or all segments fit within budget and caps).
  LP > density_greedy   when constraints are active — density_greedy violates
                        at least one of C1 or C2.

Expected latency: < 1 ms for S ≤ 1 000 segments (HiGHS back-end, CPU).
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
    threshold    : float

    Returns
    -------
    list of dicts — each with keys: start, end, length, score
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
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_triage_metrics(
    segments: list[dict],
    x: np.ndarray,
    budget_steps: int,
) -> dict:
    """Compute objective, budget usage, and coverage for one triage solution."""
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


def _count_floor_violations(
    x: np.ndarray,
    top_k_idx: np.ndarray,
    floor: float,
) -> int:
    """Count priority segments whose allocation falls below the coverage floor."""
    return int(sum(1 for k in top_k_idx if x[k] < floor - 1e-6))


def _density_greedy_fill(
    scores: np.ndarray,
    lengths: np.ndarray,
    budget_steps: int,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Density-greedy fill respecting [lb, ub] bounds.

    Used as LP fallback.  Starts from the lower-bound allocations, then fills
    remaining budget by density (score/length) descending.
    """
    n = len(scores)
    x = lb.copy()
    remaining = float(budget_steps) - float((lb * lengths).sum())
    remaining = max(remaining, 0.0)

    density = scores / np.maximum(lengths, 1.0)
    order   = np.argsort(-density)

    for idx in order:
        if remaining <= 0.0:
            break
        headroom = ub[idx] - x[idx]
        if headroom <= 1e-9:
            continue
        affordable = min(headroom, remaining / max(lengths[idx], 1.0))
        x[idx]    += affordable
        remaining -= affordable * lengths[idx]

    return x


# ---------------------------------------------------------------------------
# LP solver
# ---------------------------------------------------------------------------

def lp_triage(
    point_scores: np.ndarray,
    threshold: float,
    budget_fraction: float = 0.10,
    min_coverage_floor: float = 0.50,
    n_priority: int = 2,
    per_segment_cap: float = 0.25,
) -> tuple[list[dict], np.ndarray, bool]:
    """Constrained LP triage of anomaly segments.

    Solves a fractional knapsack LP with two operational constraints beyond the
    basic budget limit:

      C1 — Per-segment budget cap  (Constraint 1)
           length_s · x_s ≤ per_segment_cap · B   for all s
           No single segment may absorb more than `per_segment_cap` fraction of
           the total budget.  Modelled as explicit A_ub rows so the constraint
           matrix is visible to the solver.

      C2 — Minimum coverage floor  (Constraint 2)
           x_s ≥ min_coverage_floor   for the top-n_priority segments by score
           High-severity segments must receive at least this inspection fraction,
           even if they have low anomaly density (score/length ratio).

    These constraints make the problem non-trivially different from unconstrained
    density-greedy: density_greedy cannot enforce a floor on low-density priority
    segments and does not inherently cap per-segment consumption.

    Parameters
    ----------
    point_scores        : np.ndarray, shape (T,)
    threshold           : float — deployment threshold (99th pct of train errors)
    budget_fraction     : float — fraction of T to allocate (default 10 %)
    min_coverage_floor  : float — minimum x_s for top priority segments (default 0.5)
    n_priority          : int   — how many top-score segments to apply floor to
    per_segment_cap     : float — max fraction of budget one segment may use (default 0.25)

    Returns
    -------
    segments      : list of S dicts (start, end, length, score)
    x             : np.ndarray shape (S,) — constrained LP-optimal fractions
    solver_success: bool — True only when HiGHS found the optimum;
                    False when the fallback density-greedy ran instead
    """
    budget_steps = max(1, int(budget_fraction * len(point_scores)))
    segments     = extract_anomaly_candidates(point_scores, threshold)

    if not segments:
        return [], np.array([], dtype=np.float64), True

    n       = len(segments)
    scores  = np.array([seg["score"]  for seg in segments], dtype=np.float64)
    lengths = np.array([seg["length"] for seg in segments], dtype=np.float64)

    # ── Constraint 2 (C2): minimum coverage floor ──────────────────────────
    # Apply floor to top-k segments by score.  Reduce k until the floor
    # requirements alone fit within the budget (otherwise LP is infeasible).
    k = min(n, n_priority)
    top_k_idx = np.argsort(-scores)[:k]
    while k > 0:
        floor_cost = float(sum(min_coverage_floor * lengths[i] for i in top_k_idx[:k]))
        if floor_cost <= budget_steps:
            break
        k -= 1
    top_k_idx = top_k_idx[:k]

    lb = np.zeros(n, dtype=np.float64)
    lb[top_k_idx] = min_coverage_floor

    # ── Constraint 1 (C1): per-segment budget cap ───────────────────────────
    # Upper bound from cap: x_s ≤ cap · B / length_s.
    cap_steps = float(per_segment_cap) * float(budget_steps)
    ub = np.minimum(1.0, cap_steps / np.maximum(lengths, 1.0))

    # Ensure lb ≤ ub (if cap forces ub below floor, clamp lb to ub)
    lb = np.minimum(lb, ub)
    bounds = list(zip(lb.tolist(), ub.tolist()))

    # ── Constraint matrix ────────────────────────────────────────────────────
    # Row 0     : total budget   Σ length_s · x_s ≤ B
    # Rows 1..n : per-seg cap    length_s · x_s   ≤ cap · B   (explicit A_ub rows)
    A_ub = np.vstack([
        lengths.reshape(1, -1),   # (1, n)
        np.diag(lengths),         # (n, n)
    ])  # shape (n+1, n)

    b_ub = np.concatenate([
        [float(budget_steps)],
        np.full(n, cap_steps),
    ])  # shape (n+1,)

    c = -scores  # maximise signal ↔ minimise −signal

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        x = np.clip(result.x, lb, ub)
        return segments, x, True

    # ── Fallback: density-greedy respecting bounds ──────────────────────────
    x = _density_greedy_fill(scores, lengths, budget_steps, lb, ub)
    return segments, x, False


# ---------------------------------------------------------------------------
# Naive greedy baseline (sort by raw score — deliberately suboptimal)
# ---------------------------------------------------------------------------

def naive_greedy_triage(segments: list[dict], budget_steps: int) -> np.ndarray:
    """Sort by raw score descending; fill until budget exhausted.

    This is the intuitive but suboptimal heuristic for fractional knapsack.
    It ignores score/length density, so a long segment with a high absolute
    score can consume the entire budget.  It also ignores the operational
    constraints (floor and cap) entirely.
    """
    if not segments:
        return np.array([], dtype=np.float64)

    n       = len(segments)
    scores  = np.array([seg["score"]  for seg in segments], dtype=np.float64)
    lengths = np.array([seg["length"] for seg in segments], dtype=np.float64)

    order     = np.argsort(-scores)
    x         = np.zeros(n, dtype=np.float64)
    remaining = float(budget_steps)

    for idx in order:
        if remaining <= 0.0:
            break
        frac      = min(1.0, remaining / max(lengths[idx], 1.0))
        x[idx]    = frac
        remaining -= frac * lengths[idx]

    return x


# ---------------------------------------------------------------------------
# Density-greedy baseline (optimal for unconstrained fractional knapsack)
# ---------------------------------------------------------------------------

def density_greedy_triage(
    segments: list[dict],
    budget_steps: int,
    per_segment_cap: float = 0.25,
) -> np.ndarray:
    """Sort by score/length density descending; fill until budget exhausted.

    This is the provably optimal algorithm for the UNCONSTRAINED fractional
    knapsack (Dantzig 1957).  It respects the per-segment cap (C1) but does
    NOT enforce the minimum-coverage floor (C2) — it has no mechanism to force
    inspection of low-density high-priority segments.

    When C2 is active, LP will outperform density_greedy because it can
    re-balance the allocation to satisfy the floor while still spending the
    remaining budget on high-density segments.

    Parameters
    ----------
    segments         : list of dicts from extract_anomaly_candidates
    budget_steps     : int
    per_segment_cap  : float — mirrors the LP's C1 cap for a fair comparison
    """
    if not segments:
        return np.array([], dtype=np.float64)

    n       = len(segments)
    scores  = np.array([seg["score"]  for seg in segments], dtype=np.float64)
    lengths = np.array([seg["length"] for seg in segments], dtype=np.float64)

    cap_steps = float(per_segment_cap) * float(budget_steps)
    ub        = np.minimum(1.0, cap_steps / np.maximum(lengths, 1.0))
    density   = scores / np.maximum(lengths, 1.0)
    order     = np.argsort(-density)

    x         = np.zeros(n, dtype=np.float64)
    remaining = float(budget_steps)

    for idx in order:
        if remaining <= 0.0:
            break
        frac      = min(ub[idx], remaining / max(lengths[idx], 1.0))
        x[idx]    = frac
        remaining -= frac * lengths[idx]

    return x


# ---------------------------------------------------------------------------
# Three-way comparison  (main entry point used by train.py)
# ---------------------------------------------------------------------------

def compare_lp_vs_greedy(
    point_scores: np.ndarray,
    threshold: float,
    budget_fraction: float = 0.10,
    min_coverage_floor: float = 0.50,
    n_priority: int = 2,
    per_segment_cap: float = 0.25,
) -> dict:
    """Run LP, density-greedy, and naive-greedy on real LSTM-AE scores.

    Returns a comparison dict with honest framing:
      LP ≈ density_greedy   when C1/C2 are not binding
      LP > density_greedy   when constraints are active

    Parameters
    ----------
    point_scores        : np.ndarray shape (T,)  — real LSTM-AE reconstruction errors
    threshold           : float — deployment threshold
    budget_fraction     : float — default 10 %
    min_coverage_floor  : float — minimum x_s for top-n_priority segments
    n_priority          : int   — number of priority segments subject to floor
    per_segment_cap     : float — max fraction of budget per segment

    Returns
    -------
    dict with keys:
        n_candidates, total_score, budget_steps, budget_fraction, constraints,
        lp, density_greedy, naive_greedy,
        lp_gain_vs_naive_pct, lp_gain_vs_density_pct,
        lp_is_optimal,
        naive_floor_violations, density_floor_violations
    """
    budget_steps = max(1, int(budget_fraction * len(point_scores)))
    segments     = extract_anomaly_candidates(point_scores, threshold)
    total_score  = float(sum(seg["score"] for seg in segments)) if segments else 0.0

    # Resolve actual k (may be reduced if floor requirements would exceed budget)
    k = min(len(segments), n_priority)
    if segments:
        scores_arr = np.array([s["score"] for s in segments])
        lengths_arr = np.array([s["length"] for s in segments])
        top_k_idx  = np.argsort(-scores_arr)[:k]
        while k > 0:
            if float(sum(min_coverage_floor * lengths_arr[i] for i in top_k_idx[:k])) <= budget_steps:
                break
            k -= 1
        top_k_idx = top_k_idx[:k]
    else:
        top_k_idx = np.array([], dtype=int)

    # ── Run all three methods ─────────────────────────────────────────────
    _, x_lp, solver_success = lp_triage(
        point_scores, threshold, budget_fraction,
        min_coverage_floor=min_coverage_floor,
        n_priority=n_priority,
        per_segment_cap=per_segment_cap,
    )
    x_density = density_greedy_triage(segments, budget_steps, per_segment_cap=per_segment_cap)
    x_naive   = naive_greedy_triage(segments, budget_steps)

    lp_metrics      = _compute_triage_metrics(segments, x_lp,      budget_steps)
    density_metrics = _compute_triage_metrics(segments, x_density, budget_steps)
    naive_metrics   = _compute_triage_metrics(segments, x_naive,   budget_steps)

    def _pct_gain(new_val, baseline):
        if baseline > 1e-9:
            return round(100.0 * (new_val - baseline) / baseline, 2)
        return 0.0

    lp_gain_vs_naive   = _pct_gain(lp_metrics["objective"], naive_metrics["objective"])
    lp_gain_vs_density = _pct_gain(lp_metrics["objective"], density_metrics["objective"])

    naive_floor_viol   = _count_floor_violations(x_naive,   top_k_idx, min_coverage_floor)
    density_floor_viol = _count_floor_violations(x_density, top_k_idx, min_coverage_floor)

    return {
        "n_candidates":    len(segments),
        "total_score":     round(total_score, 4),
        "budget_steps":    budget_steps,
        "budget_fraction": budget_fraction,
        "constraints": {
            "min_coverage_floor":    min_coverage_floor,
            "n_priority_segments":   int(k),
            "per_segment_cap":       per_segment_cap,
        },
        "lp":              lp_metrics,
        "density_greedy":  density_metrics,
        "naive_greedy":    naive_metrics,
        "lp_gain_vs_naive_pct":   lp_gain_vs_naive,
        "lp_gain_vs_density_pct": lp_gain_vs_density,
        "lp_is_optimal":          solver_success,
        "naive_floor_violations":   naive_floor_viol,
        "density_floor_violations": density_floor_viol,
    }


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility / existing tests)
# ---------------------------------------------------------------------------

def lp_triage_summary(segments: list[dict], x: np.ndarray) -> dict:
    """Human-readable summary of an LP triage solution.

    Kept for backward compatibility — prefer compare_lp_vs_greedy for new code.
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
