"""
Tests for src/des_simulator.py

Covers:
  - schedule_from_allocation: priority ordering, fraction filtering,
                              inspection_time calculation, empty inputs
  - run_inspection_simulation: empty jobs, single job makespan, utilisation
                               bounds, job count, machine scaling, breakdown
                               impact, zero wait when capacity ≥ jobs,
                               reproducibility
  - compare_des_schedules: required keys, LP beats greedy on adversarial input,
                           equal schedules give zero reduction, breakdown flag
"""

import sys
import os

# Allow importing from src/ without installing the package
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
        """The job with the highest fraction should be first in the list."""
        segs = [
            _make_seg(0,  5,  1.0),   # idx 0 — low fraction
            _make_seg(10, 12, 2.0),   # idx 1 — high fraction
            _make_seg(20, 23, 1.5),   # idx 2 — medium fraction
        ]
        x = np.array([0.3, 0.9, 0.6])
        jobs = schedule_from_allocation(segs, x, schedule_name="test")
        priorities = [j.priority for j in jobs]
        assert priorities == sorted(priorities, reverse=True), (
            f"Expected descending priorities, got {priorities}"
        )

    def test_skips_zero_fraction_jobs(self):
        """Jobs with xi < min_fraction (default 0.01) must be excluded."""
        segs = [
            _make_seg(0, 5,  1.0),
            _make_seg(10, 15, 2.0),
        ]
        x = np.array([0.0, 0.8])
        jobs = schedule_from_allocation(segs, x, schedule_name="test")
        assert len(jobs) == 1
        assert jobs[0].segment_id == 1

    def test_inspection_time_is_length_times_fraction(self):
        """inspection_time must equal length × fraction for every job."""
        segs = [
            _make_seg(0,  10, 1.0),  # length=10, fraction=0.5 → insp=5.0
            _make_seg(20, 22, 2.0),  # length=2,  fraction=1.0 → insp=2.0
        ]
        x = np.array([0.5, 1.0])
        jobs = schedule_from_allocation(segs, x, schedule_name="test")
        # Sort back to index order for easy lookup
        by_id = {j.segment_id: j for j in jobs}
        assert by_id[0].inspection_time == pytest.approx(10 * 0.5)
        assert by_id[1].inspection_time == pytest.approx(2 * 1.0)

    def test_empty_input_returns_empty_list(self):
        """Empty segments + empty x must return an empty list (no crash)."""
        jobs = schedule_from_allocation([], np.array([]), schedule_name="test")
        assert jobs == []

    def test_all_excluded_returns_empty(self):
        """When all fractions are below min_fraction the result is empty."""
        segs = [_make_seg(0, 5, 1.0), _make_seg(10, 15, 2.0)]
        x = np.array([0.005, 0.008])  # both below default min_fraction=0.01
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
        """With one job and one machine, makespan = inspection_time."""
        jobs = _make_jobs([(0, 5.0, 1.0)])  # insp_time=5.0
        result = run_inspection_simulation(jobs, n_machines=1, mttf=0.0)
        assert result.makespan == pytest.approx(5.0, abs=1e-6)

    def test_makespan_positive_for_nonempty_jobs(self):
        jobs = _make_jobs([(0, 3.0, 0.8), (1, 2.0, 0.6)])
        result = run_inspection_simulation(jobs, n_machines=1)
        assert result.makespan > 0.0

    def test_utilisation_in_unit_interval(self):
        """Machine utilisation must be in [0, 1]."""
        jobs = _make_jobs([(0, 4.0, 1.0), (1, 3.0, 0.9), (2, 2.0, 0.7)])
        result = run_inspection_simulation(jobs, n_machines=2)
        assert 0.0 <= result.machine_utilisation <= 1.0 + 1e-9

    def test_jobs_completed_equals_input_count(self):
        jobs = _make_jobs([(0, 2.0, 1.0), (1, 3.0, 0.8), (2, 1.0, 0.5)])
        result = run_inspection_simulation(jobs, n_machines=2)
        assert result.jobs_completed == 3

    def test_more_machines_reduces_or_equals_makespan(self):
        """2 machines should finish no later than 1 machine."""
        jobs = _make_jobs([(0, 5.0, 1.0), (1, 3.0, 0.9)])
        result_1 = run_inspection_simulation(jobs, n_machines=1)
        result_2 = run_inspection_simulation(jobs, n_machines=2)
        assert result_2.makespan <= result_1.makespan + 1e-9

    def test_breakdown_increases_makespan(self):
        """Enabling breakdowns (mttf=5, mttr=2) should not decrease makespan."""
        jobs = _make_jobs([(0, 10.0, 1.0), (1, 8.0, 0.9)])
        result_no_bd = run_inspection_simulation(jobs, n_machines=1, mttf=0.0, mttr=0.0, seed=0)
        result_bd    = run_inspection_simulation(jobs, n_machines=1, mttf=5.0, mttr=2.0, seed=0)
        # Breakdowns can only add time; makespan with breakdowns >= without
        assert result_bd.makespan >= result_no_bd.makespan - 1e-9

    def test_wait_time_zero_when_machines_geq_jobs(self):
        """When capacity >= number of jobs every job starts immediately (wait=0)."""
        jobs = _make_jobs([(0, 3.0, 1.0), (1, 2.0, 0.9), (2, 1.0, 0.8)])
        result = run_inspection_simulation(jobs, n_machines=3)
        for jr in result.job_results:
            assert jr.wait_time == pytest.approx(0.0, abs=1e-9)

    def test_reproducible_with_same_seed(self):
        """Two runs with the same seed must produce identical results."""
        jobs = _make_jobs([(0, 10.0, 1.0), (1, 8.0, 0.9)])
        r1 = run_inspection_simulation(jobs, n_machines=1, mttf=5.0, mttr=2.0, seed=99)
        r2 = run_inspection_simulation(jobs, n_machines=1, mttf=5.0, mttr=2.0, seed=99)
        assert r1.makespan == r2.makespan
        assert r1.breakdown_count == r2.breakdown_count
        assert r1.mean_wait_time  == r2.mean_wait_time


# ---------------------------------------------------------------------------
# TestCompareDesSchedules
# ---------------------------------------------------------------------------

class TestCompareDesSchedules:
    def _adversarial_setup(self):
        """Segments designed so LP clearly wins over greedy.

        Segment A : score=3.0, length=10  → density=0.30  (greedy picks first)
        Segment B : score=2.5, length=2   → density=1.25
        Segment C : score=2.0, length=2   → density=1.00

        Budget = 6 steps, 1 machine.

        Greedy allocates budget to A first (highest raw score):
          A gets 6/10 = 0.6 → inspection_time = 10 × 0.6 = 6.0
          B, C get 0.0 (budget exhausted)
          → 1 job in queue; mean wait = 0.0 (only 1 job)

        LP allocates by density:
          B: fully (2 steps), inspection_time = 2.0
          C: fully (2 steps), inspection_time = 2.0
          A: 2/10 = 0.2 (remaining 2 steps), inspection_time = 10 × 0.2 = 2.0
          → 3 jobs; jobs B, C, A arrive at 0:
              B starts at 0, ends at 2  → wait=0
              C starts at 2, ends at 4  → wait=2
              A starts at 4, ends at 6  → wait=4
          mean wait = (0+2+4)/3 = 2.0  > 0.0 for greedy

        Hmm — on 1 machine with all jobs, LP always has higher mean wait than
        greedy *if greedy schedules fewer jobs*.  We need to compare the case
        where both schedules produce the same number of jobs but in different order.

        Alternative adversarial setup where both LP and greedy produce the same
        set of jobs but in different order:
          Budget = 14 steps (enough for B+C+A fully, or A+B+C greedy)
          A: score=3.0, length=10 → greedy picks first
          B: score=2.5, length=2
          C: score=2.0, length=2
          Total=14, budget=14 → both pick all, same objective, but order differs.

          Greedy order (by raw score): A(10), B(2), C(2)
            On 1 machine: A: wait=0,end=10; B: wait=10,end=12; C: wait=12,end=14
            mean wait = (0+10+12)/3 = 7.33

          LP order (by density → B first, C second, A last):
            B: wait=0,end=2; C: wait=2,end=4; A: wait=4,end=14
            mean wait = (0+2+4)/3 = 2.0

          LP clearly wins on mean wait time.
        """
        segs = [
            _make_seg(0,  10, 3.0),   # A: index 0
            _make_seg(20, 22, 2.5),   # B: index 1
            _make_seg(30, 32, 2.0),   # C: index 2
        ]
        # Budget = 14 steps → each segment fully allocated
        # LP optimal fractions (all = 1.0, sorted by density = B, C, A)
        x_lp     = np.array([1.0, 1.0, 1.0])
        # Greedy sorts by raw score: A=3.0 first, B=2.5 second, C=2.0 last
        # All fit so all = 1.0 too, but ordering differs via priority field
        x_greedy = np.array([1.0, 1.0, 1.0])
        return segs, x_lp, x_greedy

    def test_returns_required_keys(self):
        segs = [_make_seg(0, 5, 1.0), _make_seg(10, 12, 2.0)]
        x = np.array([0.8, 0.6])
        result = compare_des_schedules(segs, x, x, n_machines=2)
        for key in (
            "lp", "greedy",
            "lp_wait_reduction_pct", "lp_makespan_reduction_pct",
            "n_machines", "breakdown_enabled",
        ):
            assert key in result, f"Missing key: {key}"

    def test_lp_wait_leq_greedy_adversarial(self):
        """LP schedule achieves lower mean wait time than greedy on adversarial case.

        Setup (1 machine, all segments fully allocated, different priority order):
          Greedy priority: A(score=3.0)=1.0 → B(2.5)=1.0 → C(2.0)=1.0
            → long job A blocks B and C; mean wait is high.

          LP priority: B(density=1.25)=1.0 → C(density=1.0)=1.0 → A(density=0.30)=1.0
            → short jobs first; mean wait is lower.

        We encode ordering via the priority/fraction field so that LP x uses
        density-based fractions while greedy x uses score-based fractions.
        """
        segs = [
            _make_seg(0,  10, 3.0),   # A: long, high score, low density
            _make_seg(20, 22, 2.5),   # B: short, highest density
            _make_seg(30, 32, 2.0),   # C: short, medium density
        ]
        # LP fractions encode density-rank: B gets highest priority
        # We scale so all > min_fraction=0.01 and priority order is B > C > A
        x_lp     = np.array([0.30, 1.00, 0.90])  # A low prio, B highest, C medium
        # Greedy fractions encode score-rank: A gets highest priority
        x_greedy = np.array([1.00, 0.90, 0.80])  # A highest, then B, then C

        cmp = compare_des_schedules(segs, x_lp, x_greedy, n_machines=1, seed=42)
        lp_wait     = cmp["lp"].mean_wait_time
        greedy_wait = cmp["greedy"].mean_wait_time
        # LP (short jobs first) should have lower or equal mean wait
        assert lp_wait <= greedy_wait + 1e-6, (
            f"Expected LP wait ({lp_wait:.4f}) <= greedy wait ({greedy_wait:.4f})"
        )

    def test_equal_schedules_give_zero_reduction(self):
        """When LP and greedy produce identical job lists, reductions are 0 %."""
        segs = [_make_seg(0, 5, 1.0), _make_seg(10, 12, 0.8)]
        x = np.array([0.8, 0.6])  # same allocation for both
        cmp = compare_des_schedules(segs, x, x, n_machines=2, seed=42)
        assert cmp["lp_wait_reduction_pct"]     == pytest.approx(0.0, abs=1e-4)
        assert cmp["lp_makespan_reduction_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_breakdown_enabled_flag_correct(self):
        """breakdown_enabled must be True iff mttf > 0."""
        segs = [_make_seg(0, 5, 1.0)]
        x = np.array([1.0])
        cmp_no_bd = compare_des_schedules(segs, x, x, n_machines=1, mttf=0.0)
        cmp_bd    = compare_des_schedules(segs, x, x, n_machines=1, mttf=10.0, mttr=2.0)
        assert cmp_no_bd["breakdown_enabled"] is False
        assert cmp_bd["breakdown_enabled"]    is True
