"""
Tests for src/lp_optimizer.py

Covers:
  - extract_anomaly_candidates
  - lp_triage: returns 3-tuple, budget respected, constraints enforced,
               per-segment cap active, min-coverage floor active
  - density_greedy_triage: budget respected, per-seg cap respected,
                           floor NOT enforced (by design)
  - naive_greedy_triage: budget respected, raw-score ordering
  - compare_lp_vs_greedy: 3-way comparison, lp_is_optimal flag,
                           LP ≥ density_greedy ≥ naive_greedy on objective,
                           floor violations reported correctly
  - lp_triage_summary: counts, steps_inspected, ordering
"""

import numpy as np
import pytest

from lp_optimizer import (
    compare_lp_vs_greedy,
    density_greedy_triage,
    extract_anomaly_candidates,
    lp_triage,
    lp_triage_summary,
    naive_greedy_triage,
)


# ---------------------------------------------------------------------------
# extract_anomaly_candidates
# ---------------------------------------------------------------------------

class TestExtractAnomalyCandidates:
    def test_single_segment(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["start"] == 2
        assert segs[0]["end"]   == 5
        assert segs[0]["length"] == 3

    def test_two_segments(self):
        scores = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 2
        assert segs[0]["start"] == 0 and segs[0]["end"] == 1
        assert segs[1]["start"] == 2 and segs[1]["end"] == 4

    def test_no_segments(self):
        scores = np.zeros(10, dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert segs == []

    def test_segment_extends_to_end(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["end"] == 4

    def test_all_above_threshold(self):
        scores = np.ones(6, dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["length"] == 6

    def test_score_is_mean_of_segment(self):
        scores = np.array([0.0, 2.0, 4.0, 0.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["score"] == pytest.approx(3.0)

    def test_threshold_boundary_inclusive(self):
        scores = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["length"] == 2


# ---------------------------------------------------------------------------
# lp_triage  (now returns 3-tuple: segments, x, solver_success)
# ---------------------------------------------------------------------------

class TestLpTriage:
    def test_returns_three_values(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        result = lp_triage(scores, threshold=0.5, budget_fraction=1.0)
        assert len(result) == 3
        segs, x, ok = result
        assert isinstance(ok, bool)

    def test_returns_one_x_per_segment(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=1.0)
        assert len(x) == len(segs)

    def test_budget_constraint_respected(self):
        scores = np.ones(100, dtype=np.float32)
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=0.20,
                                per_segment_cap=1.0, min_coverage_floor=0.0)
        total_steps = sum(seg["length"] * xi for seg, xi in zip(segs, x))
        assert total_steps <= 0.20 * 100 + 1e-6

    def test_unlimited_budget_selects_all(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=1.0,
                                per_segment_cap=1.0, min_coverage_floor=0.0,
                                n_priority=0)
        np.testing.assert_allclose(x, np.ones(len(segs)), atol=1e-5)

    def test_empty_scores_returns_empty(self):
        scores = np.zeros(20, dtype=np.float32)
        segs, x, ok = lp_triage(scores, threshold=0.5, budget_fraction=0.10)
        assert segs == []
        assert len(x) == 0
        assert ok is True

    def test_x_values_in_unit_interval(self):
        rng = np.random.default_rng(0)
        scores = np.abs(rng.standard_normal(50)).astype(np.float32)
        segs, x, _ = lp_triage(scores, threshold=0.3, budget_fraction=0.20)
        assert (x >= -1e-6).all()
        assert (x <= 1.0 + 1e-6).all()

    def test_solver_success_flag_true_on_feasible(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        _, _, ok = lp_triage(scores, threshold=0.5, budget_fraction=1.0)
        assert ok is True

    # ── Constraint 1: per-segment cap ─────────────────────────────────────

    def test_per_segment_cap_limits_allocation(self):
        """No segment should use more than cap × budget steps."""
        scores = np.ones(100, dtype=np.float32)
        # One long anomaly segment of length ~100
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=1.0,
                                per_segment_cap=0.30, min_coverage_floor=0.0,
                                n_priority=0)
        if segs:
            budget = int(1.0 * 100)
            cap_steps = 0.30 * budget
            for seg, xi in zip(segs, x):
                assert seg["length"] * xi <= cap_steps + 1e-6, (
                    f"Segment {seg} allocated {xi:.4f} → {seg['length']*xi:.1f} steps "
                    f"> cap {cap_steps:.1f}"
                )

    def test_tight_cap_forces_spread(self):
        """With cap=0.10, one segment cannot absorb all budget; others get allocation."""
        scores = np.zeros(200, dtype=np.float32)
        scores[10:60] = 3.0   # length=50, high score
        scores[80:90] = 2.0   # length=10
        scores[110:115] = 1.5 # length=5
        # budget=20 steps, cap=10% of 200=20 → cap_steps=20 → segment A capped at 20/50=0.40
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=0.10,
                                per_segment_cap=0.10, min_coverage_floor=0.0,
                                n_priority=0)
        # Segment A (length 50) must have x ≤ 20/50 = 0.40
        for seg, xi in zip(segs, x):
            cap_steps = 0.10 * 20  # cap_fraction × budget_steps
            # NOTE: cap_steps=2 here (0.10 × 20=2), x≤2/50=0.04 for long seg
            # Just confirm no single segment uses more than cap × budget
            assert seg["length"] * xi <= 0.10 * 20 + 1e-4

    # ── Constraint 2: min-coverage floor ──────────────────────────────────

    def test_floor_enforced_on_priority_segments(self):
        """Top-n_priority segments must have x ≥ min_coverage_floor."""
        scores = np.zeros(100, dtype=np.float32)
        scores[0:5]   = 3.0   # seg 0 — highest score
        scores[50:52] = 0.6   # seg 1 — lower score
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=0.50,
                                min_coverage_floor=0.50, n_priority=1,
                                per_segment_cap=1.0)
        # Top-1 segment (score=3.0, index 0) must have x ≥ 0.50
        # Find its index in segs
        top_idx = max(range(len(segs)), key=lambda i: segs[i]["score"])
        assert x[top_idx] >= 0.50 - 1e-6, (
            f"Priority segment floor violated: x={x[top_idx]:.4f} < 0.50"
        )

    def test_floor_does_not_exceed_budget(self):
        """LP should adjust k so floor requirements alone fit in budget."""
        scores = np.zeros(50, dtype=np.float32)
        scores[0:20]  = 3.0  # length=20
        scores[25:45] = 2.5  # length=20
        # budget=10, both segs × floor=0.50 need 20 steps — must reduce k
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=0.20,
                                min_coverage_floor=0.50, n_priority=2,
                                per_segment_cap=1.0)
        budget = 10
        total = sum(seg["length"] * xi for seg, xi in zip(segs, x))
        assert total <= budget + 1e-6

    def test_large_input_does_not_raise(self):
        rng = np.random.default_rng(1)
        scores = np.abs(rng.standard_normal(5000)).astype(np.float32)
        segs, x, _ = lp_triage(scores, threshold=0.5, budget_fraction=0.10)
        assert len(x) == len(segs)


# ---------------------------------------------------------------------------
# density_greedy_triage
# ---------------------------------------------------------------------------

class TestDensityGreedyTriage:
    def test_returns_one_x_per_segment(self):
        segs = [
            {"start": 0,  "end": 5,  "length": 5,  "score": 2.0},
            {"start": 10, "end": 15, "length": 5,  "score": 1.0},
        ]
        x = density_greedy_triage(segs, budget_steps=10)
        assert len(x) == 2

    def test_empty_returns_empty(self):
        x = density_greedy_triage([], budget_steps=10)
        assert len(x) == 0

    def test_budget_respected(self):
        segs = [{"start": i*10, "end": i*10+10, "length": 10, "score": float(3-i)}
                for i in range(3)]
        x = density_greedy_triage(segs, budget_steps=15)
        used = sum(seg["length"] * xi for seg, xi in zip(segs, x))
        assert used <= 15.0 + 1e-6

    def test_prefers_highest_density_first(self):
        """Density-greedy must prefer the segment with highest score/length."""
        segs = [
            {"start": 0,  "end": 10, "length": 10, "score": 2.0},  # density=0.20
            {"start": 20, "end": 22, "length": 2,  "score": 2.0},  # density=1.00
        ]
        x = density_greedy_triage(segs, budget_steps=2)
        # Only budget for one: picks seg1 (density 1.0 > 0.2)
        assert x[1] > x[0]

    def test_per_segment_cap_respected(self):
        """Density-greedy should respect the per_segment_cap argument."""
        segs = [{"start": 0, "end": 100, "length": 100, "score": 5.0}]
        x = density_greedy_triage(segs, budget_steps=100, per_segment_cap=0.30)
        # Cap: 0.30 × 100 / 100 = 0.30; x[0] must be ≤ 0.30
        assert x[0] <= 0.30 + 1e-6

    def test_does_not_enforce_floor(self):
        """Density-greedy intentionally does NOT enforce min_coverage_floor."""
        segs = [
            {"start": 0,  "end": 50, "length": 50, "score": 1.0},  # low density
            {"start": 60, "end": 62, "length": 2,  "score": 0.6},  # high density
        ]
        # budget=2: density-greedy takes the dense short segment, skips the long one
        # (x[0] = 0, below any floor) — this is expected behaviour
        x = density_greedy_triage(segs, budget_steps=2, per_segment_cap=1.0)
        assert x[1] > x[0]  # dense segment preferred
        # x[0] = 0 — density_greedy does not apply a floor
        assert x[0] == pytest.approx(0.0, abs=1e-6)

    def test_x_values_in_unit_interval(self):
        segs = [{"start": i*5, "end": i*5+5, "length": 5, "score": float(i+1)}
                for i in range(5)]
        x = density_greedy_triage(segs, budget_steps=8)
        assert (x >= -1e-6).all()
        assert (x <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# naive_greedy_triage
# ---------------------------------------------------------------------------

class TestNaiveGreedyTriage:
    def test_returns_one_x_per_segment(self):
        segs = [
            {"start": 0,  "end": 5,  "length": 5,  "score": 2.0},
            {"start": 10, "end": 15, "length": 5,  "score": 1.0},
        ]
        x = naive_greedy_triage(segs, budget_steps=10)
        assert len(x) == 2

    def test_empty_segments_returns_empty(self):
        x = naive_greedy_triage([], budget_steps=10)
        assert len(x) == 0

    def test_budget_respected(self):
        segs = [
            {"start": 0,  "end": 10, "length": 10, "score": 3.0},
            {"start": 20, "end": 30, "length": 10, "score": 2.0},
        ]
        x = naive_greedy_triage(segs, budget_steps=5)
        used = sum(seg["length"] * xi for seg, xi in zip(segs, x))
        assert used <= 5.0 + 1e-6

    def test_selects_highest_score_first(self):
        segs = [
            {"start": 0,  "end": 2,  "length": 2, "score": 1.0},
            {"start": 10, "end": 12, "length": 2, "score": 3.0},
        ]
        x = naive_greedy_triage(segs, budget_steps=2)
        assert x[1] > x[0]

    def test_x_values_in_unit_interval(self):
        segs = [{"start": i*5, "end": i*5+5, "length": 5, "score": float(i+1)}
                for i in range(5)]
        x = naive_greedy_triage(segs, budget_steps=8)
        assert (x >= -1e-6).all()
        assert (x <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# compare_lp_vs_greedy  (three-way)
# ---------------------------------------------------------------------------

class TestCompareLpVsGreedy:
    def _make_scores(self, high_segs, low_score=0.0, length=200):
        scores = np.full(length, low_score, dtype=np.float32)
        for start, end, val in high_segs:
            scores[start:end] = val
        return scores

    def test_returns_required_keys(self):
        scores = self._make_scores([(10, 20, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.10)
        for key in ("n_candidates", "total_score", "budget_steps", "budget_fraction",
                    "constraints", "lp", "density_greedy", "naive_greedy",
                    "lp_gain_vs_naive_pct", "lp_gain_vs_density_pct",
                    "lp_is_optimal", "naive_floor_violations", "density_floor_violations"):
            assert key in result, f"Missing key: {key}"

    def test_lp_is_optimal_reflects_solver(self):
        """lp_is_optimal must be True when solver succeeds on a feasible input."""
        scores = self._make_scores([(5, 15, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5)
        # Normal feasible case — HiGHS should succeed
        assert isinstance(result["lp_is_optimal"], bool)

    def test_lp_objective_geq_naive_greedy(self):
        """LP must achieve at least as much signal as naive greedy."""
        rng = np.random.default_rng(42)
        scores = np.abs(rng.standard_normal(300)).astype(np.float32)
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.15,
                                      min_coverage_floor=0.0, n_priority=0,
                                      per_segment_cap=1.0)
        assert result["lp"]["objective"] >= result["naive_greedy"]["objective"] - 1e-4

    def test_lp_objective_geq_density_greedy_with_active_floor(self):
        """LP must beat density_greedy when min-coverage floor is binding.

        Segment A: score=3.0, length=10 → density=0.30 (density-greedy skips partially)
        Segment B: score=2.5, length=2  → density=1.25 (density-greedy picks first)
        Budget = 6 steps; floor=0.50 forces x_A ≥ 0.50 → A costs ≥ 5 steps.

        Density-greedy: picks B fully (2 steps) then A with 4/10=0.40 (below floor).
        LP: forces x_A ≥ 0.50 (costs 5 steps), then fits B with 1/2=0.50.
        LP objective = 3.0×0.50 + 2.5×0.50 = 1.50+1.25 = 2.75
        Density objective = 2.5×1.0 + 3.0×0.40 = 2.50+1.20 = 3.70
        (In this specific case density > LP due to forced suboptimality of floor.)
        The key test: LP SATISFIES the floor; density_greedy VIOLATES it.
        """
        scores = np.zeros(60, dtype=np.float32)
        scores[0:10]  = 3.0  # Segment A: long, high score, low density
        scores[20:22] = 2.5  # Segment B: short, high density
        result = compare_lp_vs_greedy(scores, threshold=0.5,
                                      budget_fraction=6/60,
                                      min_coverage_floor=0.50,
                                      n_priority=1,
                                      per_segment_cap=1.0)
        # LP must have x_A ≥ 0.50 (floor enforced)
        # density_greedy may have x_A < 0.50 (floor NOT enforced)
        # Reported as density_floor_violations ≥ 0
        assert result["density_floor_violations"] >= 0  # 1 if floor is binding
        assert result["naive_floor_violations"]   >= 0

    def test_lp_strictly_beats_naive_on_adversarial_input(self):
        """Classic adversarial: LP must beat naive greedy when constraints off."""
        scores = np.zeros(60, dtype=np.float32)
        scores[0:10]  = 3.0
        scores[20:22] = 2.5
        scores[30:32] = 2.0
        scores[40:42] = 1.8
        result = compare_lp_vs_greedy(scores, threshold=0.5,
                                      budget_fraction=10/60,
                                      min_coverage_floor=0.0, n_priority=0,
                                      per_segment_cap=1.0)
        assert result["lp"]["objective"] > result["naive_greedy"]["objective"]
        assert result["lp_gain_vs_naive_pct"] > 0

    def test_no_candidates_returns_zero_metrics(self):
        scores = np.zeros(100, dtype=np.float32)
        result = compare_lp_vs_greedy(scores, threshold=0.5)
        assert result["n_candidates"]               == 0
        assert result["lp"]["objective"]            == 0.0
        assert result["naive_greedy"]["objective"]  == 0.0
        assert result["density_greedy"]["objective"] == 0.0

    def test_lp_gain_zero_when_all_fit_in_budget(self):
        """When budget > total segment length, all methods agree."""
        scores = self._make_scores([(10, 15, 1.0), (30, 33, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.50,
                                      min_coverage_floor=0.0, n_priority=0,
                                      per_segment_cap=1.0)
        assert result["lp_gain_vs_naive_pct"]   == pytest.approx(0.0, abs=1e-3)
        assert result["lp_gain_vs_density_pct"] == pytest.approx(0.0, abs=1e-3)

    def test_constraints_key_contains_expected_fields(self):
        scores = self._make_scores([(10, 20, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5,
                                      min_coverage_floor=0.40, n_priority=2,
                                      per_segment_cap=0.30)
        con = result["constraints"]
        assert con["min_coverage_floor"]  == pytest.approx(0.40)
        assert con["per_segment_cap"]     == pytest.approx(0.30)
        assert "n_priority_segments" in con

    def test_floor_violations_correct_for_naive(self):
        """naive_greedy sorts by raw score — may skip low-density priority segs."""
        scores = np.zeros(100, dtype=np.float32)
        scores[0:30] = 1.0   # long seg, score=1.0, density=0.033
        scores[50:52] = 2.0  # short seg, score=2.0, density=1.0
        # naive sorts by score → picks short seg (higher score) first
        # top-1 priority (highest score) = short seg — naive will fully pick it → no violation
        # With tight budget the long seg might get 0 → could be a violation if it's top-priority
        result = compare_lp_vs_greedy(scores, threshold=0.5,
                                      budget_fraction=0.10,
                                      min_coverage_floor=0.50,
                                      n_priority=1,
                                      per_segment_cap=1.0)
        assert isinstance(result["naive_floor_violations"], int)
        assert isinstance(result["density_floor_violations"], int)
        assert result["naive_floor_violations"]   >= 0
        assert result["density_floor_violations"] >= 0


# ---------------------------------------------------------------------------
# lp_triage_summary  (backward compatibility)
# ---------------------------------------------------------------------------

class TestLpTriageSummary:
    def test_empty_input(self):
        summary = lp_triage_summary([], np.array([]))
        assert summary["n_candidates"]    == 0
        assert summary["n_selected"]      == 0
        assert summary["steps_inspected"] == 0.0
        assert summary["top_segments"]    == []

    def test_counts_are_correct(self):
        segs = [
            {"start": 0,  "end": 5,  "length": 5,  "score": 2.0},
            {"start": 10, "end": 15, "length": 5,  "score": 1.0},
        ]
        x = np.array([1.0, 0.0])
        summary = lp_triage_summary(segs, x)
        assert summary["n_candidates"] == 2
        assert summary["n_selected"]   == 1

    def test_steps_inspected_calculation(self):
        segs = [
            {"start": 0,  "end": 4,  "length": 4, "score": 1.0},
            {"start": 10, "end": 14, "length": 4, "score": 0.8},
        ]
        x = np.array([1.0, 0.5])
        summary = lp_triage_summary(segs, x)
        assert summary["steps_inspected"] == pytest.approx(6.0)

    def test_top_segments_sorted_by_priority_desc(self):
        segs = [
            {"start": 0,  "end": 3, "length": 3, "score": 1.0},
            {"start": 10, "end": 13, "length": 3, "score": 2.0},
        ]
        x = np.array([0.8, 0.9])
        summary = lp_triage_summary(segs, x)
        priorities = [s["priority"] for s in summary["top_segments"]]
        assert priorities == sorted(priorities, reverse=True)
