"""
Tests for src/lp_optimizer.py

Covers:
  - extract_anomaly_candidates: single segment, multiple segments, no segments,
                                 segment at end of array, all-above threshold
  - lp_triage: budget constraint respected, unlimited budget → all ones,
               zero budget → all zeros, empty scores, single segment,
               higher-score segment preferred over lower-score one
  - lp_triage_summary: counts, steps_inspected, top_segments order
"""

import numpy as np
import pytest

from lp_optimizer import (
    compare_lp_vs_greedy,
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
        """Segment that runs to the last index must be captured."""
        scores = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["end"] == 4   # exclusive end = len(scores)

    def test_all_above_threshold(self):
        scores = np.ones(6, dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["length"] == 6

    def test_score_is_mean_of_segment(self):
        scores = np.array([0.0, 2.0, 4.0, 0.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["score"] == pytest.approx(3.0)  # mean(2, 4)

    def test_threshold_boundary_inclusive(self):
        """Points exactly at the threshold should be included."""
        scores = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
        segs = extract_anomaly_candidates(scores, threshold=0.5)
        assert len(segs) == 1
        assert segs[0]["length"] == 2


# ---------------------------------------------------------------------------
# lp_triage
# ---------------------------------------------------------------------------

class TestLpTriage:
    def _uniform_scores(self, length=100, high_val=1.0, low_val=0.0):
        """Array with first half high, second half low."""
        scores = np.full(length, low_val, dtype=np.float32)
        scores[:length // 2] = high_val
        return scores

    def test_returns_one_x_per_segment(self):
        scores = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=1.0)
        assert len(x) == len(segs)

    def test_budget_constraint_respected(self):
        """Total steps allocated must not exceed the budget."""
        scores = np.ones(100, dtype=np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=0.20)
        total_steps = sum(seg["length"] * xi for seg, xi in zip(segs, x))
        budget = 0.20 * 100
        assert total_steps <= budget + 1e-6   # small tolerance for float rounding

    def test_unlimited_budget_selects_all(self):
        """With budget ≥ total segment length every x_s should be 1."""
        scores = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=1.0)
        np.testing.assert_allclose(x, np.ones(len(segs)), atol=1e-5)

    def test_zero_budget_gives_all_zeros(self):
        """budget_fraction → 0 means budget_steps=1 (clipped), but only the
        highest score/length segment gets priority; overall allocated steps ≤ 1."""
        scores = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=0.0)
        total_steps = sum(seg["length"] * xi for seg, xi in zip(segs, x))
        assert total_steps <= 1.0 + 1e-6

    def test_empty_scores_returns_empty(self):
        scores = np.zeros(20, dtype=np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=0.10)
        assert segs == []
        assert len(x) == 0

    def test_single_segment_within_budget(self):
        scores = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=1.0)
        assert len(segs) == 1
        assert x[0] == pytest.approx(1.0, abs=1e-5)

    def test_prefers_higher_score_segment(self):
        """With a tight budget the LP should favour the segment with higher score."""
        # Two segments: one short+high score, one long+low score
        # budget = 5 steps → can only fully cover one
        scores = np.zeros(40, dtype=np.float32)
        scores[0:5]   = 2.0   # high score, length=5
        scores[20:30] = 0.6   # lower score, length=10
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=5 / 40)
        # segment 0 has score=2.0, segment 1 has score=0.6
        # LP should allocate budget to segment 0 first
        assert x[0] >= x[1]

    def test_x_values_in_unit_interval(self):
        scores = np.random.default_rng(0).standard_normal(50).astype(np.float32)
        scores = np.abs(scores)
        segs, x = lp_triage(scores, threshold=0.3, budget_fraction=0.20)
        assert (x >= -1e-6).all()
        assert (x <= 1.0 + 1e-6).all()

    def test_large_input_does_not_raise(self):
        rng = np.random.default_rng(1)
        scores = np.abs(rng.standard_normal(5000)).astype(np.float32)
        segs, x = lp_triage(scores, threshold=0.5, budget_fraction=0.10)
        assert len(x) == len(segs)


# ---------------------------------------------------------------------------
# lp_triage_summary
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
            {"start": 0, "end": 5,  "length": 5,  "score": 2.0},
            {"start": 10, "end": 15, "length": 5, "score": 1.0},
        ]
        x = np.array([1.0, 0.0])
        summary = lp_triage_summary(segs, x)
        assert summary["n_candidates"] == 2
        assert summary["n_selected"]   == 1   # only x > 0.5

    def test_steps_inspected_calculation(self):
        segs = [
            {"start": 0, "end": 4,  "length": 4,  "score": 1.0},
            {"start": 10, "end": 14, "length": 4, "score": 0.8},
        ]
        x = np.array([1.0, 0.5])
        summary = lp_triage_summary(segs, x)
        # 4*1.0 + 4*0.5 = 6.0
        assert summary["steps_inspected"] == pytest.approx(6.0)

    def test_top_segments_sorted_by_priority_desc(self):
        segs = [
            {"start": 0,  "end": 3,  "length": 3, "score": 1.0},
            {"start": 10, "end": 13, "length": 3, "score": 2.0},
        ]
        x = np.array([0.8, 0.9])
        summary = lp_triage_summary(segs, x)
        priorities = [s["priority"] for s in summary["top_segments"]]
        assert priorities == sorted(priorities, reverse=True)

    def test_top_segments_only_includes_high_priority(self):
        """Only segments with x > 0.5 appear in top_segments."""
        segs = [
            {"start": 0,  "end": 5,  "length": 5, "score": 1.0},
            {"start": 10, "end": 15, "length": 5, "score": 0.5},
        ]
        x = np.array([0.9, 0.3])   # second segment below 0.5 threshold
        summary = lp_triage_summary(segs, x)
        assert summary["n_selected"] == 1
        assert summary["top_segments"][0]["start"] == 0


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
        """Naive greedy picks the segment with higher raw score first."""
        segs = [
            {"start": 0,  "end": 2,  "length": 2, "score": 1.0},  # lower score
            {"start": 10, "end": 12, "length": 2, "score": 3.0},  # higher score
        ]
        x = naive_greedy_triage(segs, budget_steps=2)
        # Only budget for one segment — should pick index 1 (score=3.0)
        assert x[1] > x[0]

    def test_x_values_in_unit_interval(self):
        segs = [{"start": i*5, "end": i*5+5, "length": 5, "score": float(i+1)}
                for i in range(5)]
        x = naive_greedy_triage(segs, budget_steps=8)
        assert (x >= -1e-6).all()
        assert (x <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# compare_lp_vs_greedy
# ---------------------------------------------------------------------------

class TestCompareLpVsGreedy:
    def _make_scores(self, high_segs, low_score=0.0):
        """Build a point_scores array with controllable anomaly segments."""
        scores = np.full(200, low_score, dtype=np.float32)
        for start, end, val in high_segs:
            scores[start:end] = val
        return scores

    def test_returns_required_keys(self):
        scores = self._make_scores([(10, 20, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.10)
        for key in ("n_candidates", "total_score", "budget_steps",
                    "budget_fraction", "lp", "greedy", "lp_gain_pct", "lp_is_optimal"):
            assert key in result, f"Missing key: {key}"

    def test_lp_is_optimal_always_true(self):
        scores = self._make_scores([(5, 15, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5)
        assert result["lp_is_optimal"] is True

    def test_lp_objective_geq_greedy(self):
        """LP must achieve at least as much as naive greedy on any input."""
        rng = np.random.default_rng(42)
        scores = np.abs(rng.standard_normal(300)).astype(np.float32)
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.15)
        assert result["lp"]["objective"] >= result["greedy"]["objective"] - 1e-4

    def test_lp_strictly_beats_greedy_on_adversarial_input(self):
        """Construct a case where naive greedy is provably suboptimal.

        One long segment with the highest raw score but low density, and
        several short dense segments.  With a tight budget the LP should
        prefer the dense segments while greedy wastes budget on the long one.

        Budget = 10 steps.
        Segment A: score=3.0, length=10 → density=0.30  (greedy picks first)
        Segment B: score=2.5, length=2  → density=1.25  (LP picks first)
        Segment C: score=2.0, length=2  → density=1.00
        Segment D: score=1.8, length=2  → density=0.90

        Greedy: picks A (10 steps) → objective = 3.0
        LP:     B(2) + C(2) + D(2) + 4/10 of A → objective = 2.5+2.0+1.8+1.2 = 7.5
        """
        scores = np.zeros(60, dtype=np.float32)
        scores[0:10]  = 3.0   # Segment A: long, high score, low density
        scores[20:22] = 2.5   # Segment B: short, high density
        scores[30:32] = 2.0   # Segment C
        scores[40:42] = 1.8   # Segment D
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=10/60)
        assert result["lp"]["objective"] > result["greedy"]["objective"]
        assert result["lp_gain_pct"] > 0

    def test_no_candidates_returns_zero_metrics(self):
        scores = np.zeros(100, dtype=np.float32)
        result = compare_lp_vs_greedy(scores, threshold=0.5)
        assert result["n_candidates"]      == 0
        assert result["lp"]["objective"]   == 0.0
        assert result["greedy"]["objective"] == 0.0

    def test_budget_fraction_stored_correctly(self):
        scores = self._make_scores([(10, 20, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.20)
        assert result["budget_fraction"] == pytest.approx(0.20)
        assert result["budget_steps"]    == 40   # 0.20 * 200

    def test_lp_gain_zero_when_all_fit_in_budget(self):
        """When budget > total segment length, LP and greedy are identical."""
        scores = self._make_scores([(10, 15, 1.0), (30, 33, 1.0)])
        # 8 total anomaly steps, budget = 50 % of 200 = 100 → all fit
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.50)
        assert result["lp_gain_pct"] == pytest.approx(0.0, abs=1e-3)

    def test_coverage_pct_100_when_unlimited_budget(self):
        scores = self._make_scores([(5, 10, 1.0)])
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=1.0)
        assert result["lp"]["coverage_pct"] == pytest.approx(100.0, abs=1e-3)

    def test_budget_utilization_leq_100(self):
        scores = self._make_scores([(10, 20, 1.0), (40, 50, 1.5)])
        result = compare_lp_vs_greedy(scores, threshold=0.5, budget_fraction=0.10)
        assert result["lp"]["budget_utilization_pct"]     <= 100.0 + 1e-4
        assert result["greedy"]["budget_utilization_pct"] <= 100.0 + 1e-4
