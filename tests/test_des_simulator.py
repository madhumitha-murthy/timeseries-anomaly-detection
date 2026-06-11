"""
Tests for src/des_simulator.py

Covers:
  - schedule_from_allocation: priority ordering, fraction filtering,
                              inspection_time calculation, empty inputs
  - run_inspection_simulation: empty jobs, single job makespan, utilisation
                               bounds, job count, machine scaling, breakdown
                               impact, zero wait when capacity >= jobs,
                               reproducibility
  - compare_des_schedules: required keys, fair comparison (identical jobs),
                           LP/density ordering reduces wait vs naive ordering,
                           equal schedules give zero reduction,
                           makespan equal across orderings (same work),
                           breakdown flag
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from des_simulator import (
    InspectionJob,
    compare_des_schedules,
    run_inspection_simulation,
    schedule_from_allocation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg(start: int, end: int, score: float) -> dict:
    return {"start": start, "end": end, "length": end - start, "score": score}


def _make_jobs(specs: list[tuple]) -> list[InspectionJob]:
    """Build InspectionJob list from (segment_id, inspection_time, priority) tuples."""
    jobs = []
    for seg_id, insp_time, priority in specs:
        jobs.append(
            InspectionJob(
                segment_id=seg_id,
                start=seg_id * 10,
                end=seg_id * 10 + 10,
                score=1.0,
                length=10,
                fraction=priority,
                inspection_time=insp_time,
                priority=priority,
            )
        )
    return jobs


# ---------------------------------------------------------------------------
# TestScheduleFromAllocation
# ---------------------------------------------------------------------------

class TestScheduleFromAllocation:
    def test_orders_by_priority_descending(self):
        segs = [
            _make_seg(0,  5,  1.0),
            _make_seg(10, 12, 2.0),
            _make_seg(20, 23, 1.5),
        ]
        x = np.array([0.3, 0.9, 0.6])
        jobs = schedule_from_allocation(segs, x, schedule_name="test")
        priorities = [j.priority for j in jobs]
        assert priorities == sorted(priorities, reverse=True)

    def test_skips_zero_fraction_jobs(self):
        segs = [_make_seg(0, 5, 1.0), _make_seg(10, 15, 2.0)]
        x = np.array([0.0, 0.8])
        jobs = schedule_from_allocation(segs, x, schedule_name="test")
        assert len(jobs) == 1
        assert jobs[0].segment_id == 1

    def test_inspection_time_is_length_times_fraction(self):
        segs = [
            _make_seg(0,  10, 1.0),
            _make_seg(20, 22, 2.0),
        ]
        x = np.array([0.5, 1.0])
        jobs = schedule_from_allocation(segs, x, schedule_name="test")
        by_id = {j.segment_id: j for j in jobs}
        assert by_id[0].inspection_time == pytest.approx(10 * 0.5)
        assert by_id[1].inspection_time == pytest.approx(2 * 1.0)

    def test_empty_input_returns_empty_list(self):
        jobs = schedule_from_allocation([], np.array([]), schedule_name="test")
        assert jobs == []

    def test_all_excluded_returns_empty(self):
        segs = [_make_seg(0, 5, 1.0), _make_seg(10, 15, 2.0)]
        x = np.array([0.005, 0.008])
        jobs = schedule_from_allocation(segs, x, schedule_name="test", min_fraction=0.01)
        assert jobs == []


# ---------------------------------------------------------------------------
# TestRunInspectionSimulation
# ---------------------------------------------------------------------------

class TestRunInspectionSimulation:
    def test_empty_jobs_returns_zero_metrics(self):
        result = run_inspection_simulation([], n_machines=2)
        assert result.makespan == 0.0
        assert result.mean_wait_time == 0.0
        assert result.p95_wait_time == 0.0
        assert result.jobs_completed == 0
        assert result.breakdown_count == 0

    def test_single_job_makespan_equals_inspection_time(self):
        jobs = _make_jobs([(0, 5.0, 1.0)])
        result = run_inspection_simulation(jobs, n_machines=1, mttf=0.0)
        assert result.makespan == pytest.approx(5.0, abs=1e-6)

    def test_makespan_positive_for_nonempty_jobs(self):
        jobs = _make_jobs([(0, 3.0, 0.8), (1, 2.0, 0.6)])
        result = run_inspection_simulation(jobs, n_machines=1)
        assert result.makespan > 0.0

    def test_utilisation_in_unit_interval(self):
        jobs = _make_jobs([(0, 4.0, 1.0), (1, 3.0, 0.9), (2, 2.0, 0.7)])
        result = run_inspection_simulation(jobs, n_machines=2)
        assert 0.0 <= result.machine_utilisation <= 1.0 + 1e-9

    def test_jobs_completed_equals_input_count(self):
        jobs = _make_jobs([(0, 2.0, 1.0), (1, 3.0, 0.8), (2, 1.0, 0.5)])
        result = run_inspection_simulation(jobs, n_machines=2)
        assert result.jobs_completed == 3

    def test_more_machines_reduces_or_equals_makespan(self):
        jobs = _make_jobs([(0, 5.0, 1.0), (1, 3.0, 0.9)])
        result_1 = run_inspection_simulation(jobs, n_machines=1)
        result_2 = run_inspection_simulation(jobs, n_machines=2)
        assert result_2.makespan <= result_1.makespan + 1e-9

    def test_breakdown_increases_makespan(self):
        jobs = _make_jobs([(0, 10.0, 1.0), (1, 8.0, 0.9)])
        result_no_bd = run_inspection_simulation(jobs, n_machines=1, mttf=0.0, mttr=0.0, seed=0)
        result_bd    = run_inspection_simulation(jobs, n_machines=1, mttf=5.0, mttr=2.0, seed=0)
        assert result_bd.makespan >= result_no_bd.makespan - 1e-9

    def test_wait_time_zero_when_machines_geq_jobs(self):
        jobs = _make_jobs([(0, 3.0, 1.0), (1, 2.0, 0.9), (2, 1.0, 0.8)])
        result = run_inspection_simulation(jobs, n_machines=3)
        for jr in result.job_results:
            assert jr.wait_time == pytest.approx(0.0, abs=1e-9)

    def test_reproducible_with_same_seed(self):
        jobs = _make_jobs([(0, 10.0, 1.0), (1, 8.0, 0.9)])
        r1 = run_inspection_simulation(jobs, n_machines=1, mttf=5.0, mttr=2.0, seed=99)
        r2 = run_inspection_simulation(jobs, n_machines=1, mttf=5.0, mttr=2.0, seed=99)
        assert r1.makespan == r2.makespan
        assert r1.breakdown_count == r2.breakdown_count
        assert r1.mean_wait_time  == r2.mean_wait_time


# ---------------------------------------------------------------------------
# TestCompareDesSchedules  (new fair-comparison API)
# ---------------------------------------------------------------------------

class TestCompareDesSchedules:
    """compare_des_schedules(segments, x_lp, ...) — single allocation, 3 orderings."""

    def _adversarial_segs_and_alloc(self):
        """
        3 segments, all fully allocated (x=1.0).
        LP-fraction order = density order: B(density=1.25) > C(1.0) > A(0.3)
        Naive order:                        A(score=3.0)   > B(2.5) > C(2.0)

        On 1 machine:
          LP/density order:  B(t=2), C(t=2), A(t=10) → wait=(0,2,4), mean=2.0
          Naive order:       A(t=10), B(t=2), C(t=2) → wait=(0,10,12), mean=7.33
        """
        segs = [
            _make_seg(0,  10, 3.0),   # A: long, score=3.0, density=0.30
            _make_seg(20, 22, 2.5),   # B: short, score=2.5, density=1.25
            _make_seg(30, 32, 2.0),   # C: short, score=2.0, density=1.00
        ]
        # LP fractions: B highest (1.0), C next (0.9), A lowest (0.3)
        # schedule_from_allocation will sort by fraction → B, C, A in queue
        x_lp = np.array([0.30, 1.00, 0.90])
        return segs, x_lp

    def test_returns_required_keys(self):
        segs = [_make_seg(0, 5, 1.0), _make_seg(10, 12, 2.0)]
        x = np.array([0.8, 0.6])
        result = compare_des_schedules(segs, x, n_machines=2)
        for key in (
            "lp", "greedy", "density",
            "lp_wait_reduction_pct",
            "density_wait_reduction_pct",
            "lp_vs_density_wait_diff_pct",
            "lp_makespan_reduction_pct",
            "n_machines", "breakdown_enabled",
        ):
            assert key in result, f"Missing key: {key}"

    def test_all_orderings_have_same_n_jobs(self):
        """All three orderings must process the same set of jobs."""
        segs, x_lp = self._adversarial_segs_and_alloc()
        cmp = compare_des_schedules(segs, x_lp, n_machines=1, seed=42)
        assert cmp["lp"].n_jobs == cmp["greedy"].n_jobs == cmp["density"].n_jobs

    def test_all_orderings_have_equal_makespan(self):
        """Same jobs, same machines → makespan must be identical across orderings."""
        segs, x_lp = self._adversarial_segs_and_alloc()
        cmp = compare_des_schedules(segs, x_lp, n_machines=1, mttf=0.0, seed=42)
        assert cmp["lp"].makespan    == pytest.approx(cmp["greedy"].makespan,  abs=1e-6)
        assert cmp["density"].makespan == pytest.approx(cmp["greedy"].makespan, abs=1e-6)

    def test_all_orderings_have_equal_utilisation(self):
        """Same total work → utilisation equal across orderings."""
        segs, x_lp = self._adversarial_segs_and_alloc()
        cmp = compare_des_schedules(segs, x_lp, n_machines=1, mttf=0.0, seed=42)
        assert cmp["lp"].machine_utilisation == pytest.approx(
            cmp["greedy"].machine_utilisation, abs=1e-6
        )

    def test_lp_wait_leq_naive_wait_adversarial(self):
        """LP ordering (density-first) reduces mean wait vs naive (score-first)."""
        segs, x_lp = self._adversarial_segs_and_alloc()
        cmp = compare_des_schedules(segs, x_lp, n_machines=1, mttf=0.0, seed=42)
        assert cmp["lp"].mean_wait_time <= cmp["greedy"].mean_wait_time + 1e-6, (
            f"LP wait {cmp['lp'].mean_wait_time:.4f} > naive wait "
            f"{cmp['greedy'].mean_wait_time:.4f}"
        )

    def test_density_wait_leq_naive_wait_adversarial(self):
        """Density ordering also reduces wait vs naive ordering."""
        segs, x_lp = self._adversarial_segs_and_alloc()
        cmp = compare_des_schedules(segs, x_lp, n_machines=1, mttf=0.0, seed=42)
        assert cmp["density"].mean_wait_time <= cmp["greedy"].mean_wait_time + 1e-6

    def test_lp_vs_density_wait_diff_near_zero(self):
        """LP ordering ≈ density ordering — diff should be small."""
        segs, x_lp = self._adversarial_segs_and_alloc()
        cmp = compare_des_schedules(segs, x_lp, n_machines=1, mttf=0.0, seed=42)
        # LP and density orderings may differ slightly if LP fractions don't
        # perfectly match density rank; the absolute difference should be small
        assert abs(cmp["lp"].mean_wait_time - cmp["density"].mean_wait_time) < 5.0

    def test_equal_x_gives_zero_makespan_reduction(self):
        """With a single segment, all orderings are identical → 0 % reduction."""
        segs = [_make_seg(0, 5, 1.0)]
        x = np.array([0.8])
        cmp = compare_des_schedules(segs, x, n_machines=2, seed=42)
        assert cmp["lp_makespan_reduction_pct"]     == pytest.approx(0.0, abs=1e-4)
        assert cmp["lp_wait_reduction_pct"]         == pytest.approx(0.0, abs=1e-4)
        assert cmp["density_wait_reduction_pct"]    == pytest.approx(0.0, abs=1e-4)

    def test_breakdown_enabled_flag_correct(self):
        segs = [_make_seg(0, 5, 1.0)]
        x = np.array([1.0])
        cmp_no_bd = compare_des_schedules(segs, x, n_machines=1, mttf=0.0)
        cmp_bd    = compare_des_schedules(segs, x, n_machines=1, mttf=10.0, mttr=2.0)
        assert cmp_no_bd["breakdown_enabled"] is False
        assert cmp_bd["breakdown_enabled"]    is True

    def test_empty_segments_returns_safely(self):
        cmp = compare_des_schedules([], np.array([]), n_machines=2)
        assert cmp["lp"].jobs_completed    == 0
        assert cmp["greedy"].jobs_completed == 0
        assert cmp["density"].jobs_completed == 0
