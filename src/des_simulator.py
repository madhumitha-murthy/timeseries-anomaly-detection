"""
des_simulator.py — Discrete Event Simulation of the post-LP inspection workflow.

Full pipeline
─────────────
  1. LSTM-AE detects anomaly segments
       The trained autoencoder scores every test time-step; contiguous runs of
       above-threshold scores become *candidate anomaly segments*.

  2. LP prioritises segments  (src/lp_optimizer.py)
       A fractional-knapsack LP allocates an inspection fraction x_s ∈ [0, 1]
       to each segment, maximising anomaly signal covered within a fixed budget.
       The LP provably beats naive greedy sorting by raw score when segments
       differ in length (density matters, not just absolute score).

  3. DES simulates the inspection queue  (this module)
       After the LP decides *which* segments to inspect and at *what fraction*,
       field engineers must physically carry out those inspections.  This module
       models the resulting workflow as a discrete-event simulation (SimPy):

         • N parallel inspection machines (engineers / workstations)
         • All jobs (segments) arrive at time 0, queued in priority order
         • Each job occupies one machine for  length × fraction  time units
         • Machine breakdowns occur inline during service (exponential MTTF/MTTR)
         • Key metrics: makespan, mean/p95 wait time, utilisation, throughput

Methodologically sound comparison
──────────────────────────────────
The LP allocation (which segments to inspect, at what fraction) is FIXED.
compare_des_schedules creates three queue orderings from IDENTICAL job sets:

  • LP ordering      — jobs sorted by LP fraction (density proxy) descending
                       High-density short jobs first → lower wait time
  • Naive ordering   — same jobs sorted by raw score descending
                       Long high-score job may block shorter urgent jobs
  • Density ordering — same jobs sorted by score/length density descending
                       Should ≈ LP ordering (confirms LP fractions ≈ density rank)

Because all three orderings share identical inspection_times, makespan and
utilisation are equal across orderings.  Only wait-time distribution differs.
This isolates the effect of queue discipline from the allocation decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

import simpy


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InspectionJob:
    """One anomaly segment scheduled for physical inspection.

    Parameters
    ----------
    segment_id      : int   Index into the original segments list.
    start           : int   Inclusive start time-step.
    end             : int   Exclusive end time-step.
    score           : float Mean LSTM-AE reconstruction error (anomaly density).
    length          : int   Number of time-steps in the segment.
    fraction        : float LP-allocated inspection fraction ∈ (0, 1].
    inspection_time : float Effective inspection duration = length × fraction.
    priority        : float Scheduling priority (higher → inspected first);
                            set to the LP fraction so the LP schedule is
                            automatically in priority order.
    """
    segment_id:      int
    start:           int
    end:             int
    score:           float
    length:          int
    fraction:        float
    inspection_time: float
    priority:        float


@dataclass
class JobResult:
    """Timing outcome for one completed inspection job.

    Parameters
    ----------
    segment_id      : int   Matches InspectionJob.segment_id.
    arrival_time    : float Time the job entered the queue (always 0.0 here).
    start_time      : float Time a machine became available for this job.
    end_time        : float Time the inspection (including any repairs) finished.
    wait_time       : float start_time − arrival_time.
    inspection_time : float Actual service duration (excluding wait).
    """
    segment_id:      int
    arrival_time:    float
    start_time:      float
    end_time:        float
    wait_time:       float
    inspection_time: float


@dataclass
class SimulationResult:
    """Aggregate metrics for one completed DES run.

    Parameters
    ----------
    schedule_name        : str    "lp" or "greedy" (or any label).
    n_jobs               : int    Number of jobs submitted.
    n_machines           : int    Parallel inspection capacity.
    makespan             : float  Time from start until last job finishes.
    mean_wait_time       : float  Average queue wait across all jobs.
    p95_wait_time        : float  95th-percentile queue wait.
    mean_inspection_time : float  Average active inspection duration.
    machine_utilisation  : float  Fraction of machine-time spent working ∈ [0, 1].
    throughput           : float  Jobs completed per time unit (1 / makespan × n_jobs).
    jobs_completed       : int    Number of jobs that finished.
    breakdown_count      : int    Total mid-service breakdowns that occurred.
    job_results          : list   Per-job JobResult records.
    """
    schedule_name:        str
    n_jobs:               int
    n_machines:           int
    makespan:             float
    mean_wait_time:       float
    p95_wait_time:        float
    mean_inspection_time: float
    machine_utilisation:  float
    throughput:           float
    jobs_completed:       int
    breakdown_count:      int
    job_results:          List[JobResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------

def schedule_from_allocation(
    segments: list[dict],
    x: "np.ndarray",  # noqa: F821  (imported at call-site; avoid hard dep here)
    schedule_name: str,
    min_fraction: float = 0.01,
) -> List[InspectionJob]:
    """Convert LP/greedy allocation arrays into a sorted inspection job list.

    Jobs are sorted by priority (LP fraction) descending so the most-favoured
    segments enter the queue first.  Jobs whose allocated fraction falls below
    *min_fraction* are excluded (they contribute negligible signal).

    Parameters
    ----------
    segments      : list of segment dicts from extract_anomaly_candidates
                    Keys: start, end, length, score.
    x             : np.ndarray, shape (S,)
                    Allocation fractions in [0, 1] from lp_triage or naive_greedy_triage.
    schedule_name : str
                    Label attached to each job (informational).
    min_fraction  : float, default 0.01
                    Segments with x_s < min_fraction are dropped.

    Returns
    -------
    List[InspectionJob] sorted by priority descending (highest fraction first).
    """
    jobs: List[InspectionJob] = []
    for idx, (seg, xi) in enumerate(zip(segments, x)):
        xi = float(xi)
        if xi < min_fraction:
            continue
        inspection_time = float(seg["length"]) * xi
        jobs.append(
            InspectionJob(
                segment_id=idx,
                start=int(seg["start"]),
                end=int(seg["end"]),
                score=float(seg["score"]),
                length=int(seg["length"]),
                fraction=xi,
                inspection_time=inspection_time,
                priority=xi,
            )
        )
    # Higher priority (fraction) → front of queue
    jobs.sort(key=lambda j: -j.priority)
    return jobs


# ---------------------------------------------------------------------------
# SimPy simulation
# ---------------------------------------------------------------------------

def run_inspection_simulation(
    jobs: List[InspectionJob],
    n_machines: int = 2,
    mttf: float = 0.0,
    mttr: float = 0.0,
    schedule_name: str = "unnamed",
    seed: int = 42,
) -> SimulationResult:
    """Run a SimPy discrete-event simulation of the inspection queue.

    All jobs arrive at time 0 and are processed in the order given by *jobs*
    (caller is responsible for sorting by priority before passing in).
    Machines are modelled as a single SimPy Resource with capacity *n_machines*.

    Breakdown model
    ───────────────
    When mttf > 0, each unit of service can be interrupted by a breakdown:

      while service_remaining > 0:
          draw time_to_failure ~ Exp(1/mttf)
          if ttf >= remaining:
              work for remaining; done
          else:
              work for ttf; break down
              draw repair_time ~ Exp(1/mttr)
              wait for repair; resume

    Parameters
    ----------
    jobs          : List[InspectionJob]
        Jobs sorted by priority (highest first).  An empty list is valid.
    n_machines    : int, default 2
        Number of parallel inspection machines / engineers.
    mttf          : float, default 0.0
        Mean time to failure (time units).  0 means no breakdowns.
    mttr          : float, default 0.0
        Mean time to repair (time units).  Ignored when mttf == 0.
    schedule_name : str
        Label stored in the returned SimulationResult.
    seed          : int, default 42
        RNG seed for reproducible breakdown draws.

    Returns
    -------
    SimulationResult with all aggregate metrics and per-job JobResult records.
    """
    if not jobs:
        return SimulationResult(
            schedule_name=schedule_name,
            n_jobs=0,
            n_machines=n_machines,
            makespan=0.0,
            mean_wait_time=0.0,
            p95_wait_time=0.0,
            mean_inspection_time=0.0,
            machine_utilisation=0.0,
            throughput=0.0,
            jobs_completed=0,
            breakdown_count=0,
            job_results=[],
        )

    rng = random.Random(seed)
    env = simpy.Environment()
    machines = simpy.Resource(env, capacity=n_machines)

    collected: List[JobResult] = []
    breakdown_count_holder: List[int] = [0]  # mutable int via list

    def _inspect_job(job: InspectionJob) -> None:
        """SimPy generator: queue → acquire machine → service with breakdowns."""
        arrival_time = env.now  # always 0.0 in this setup (batch arrival)

        with machines.request() as req:
            yield req
            start_time = env.now

            # --- Inline breakdown model ---
            remaining = job.inspection_time
            while remaining > 0.0:
                if mttf > 0.0:
                    ttf = rng.expovariate(1.0 / mttf)
                    if ttf >= remaining:
                        yield env.timeout(remaining)
                        remaining = 0.0
                    else:
                        yield env.timeout(ttf)
                        remaining -= ttf
                        repair_time = rng.expovariate(1.0 / mttr) if mttr > 0.0 else 0.0
                        yield env.timeout(repair_time)
                        breakdown_count_holder[0] += 1
                else:
                    yield env.timeout(remaining)
                    remaining = 0.0

            end_time = env.now
            collected.append(
                JobResult(
                    segment_id=job.segment_id,
                    arrival_time=arrival_time,
                    start_time=start_time,
                    end_time=end_time,
                    wait_time=start_time - arrival_time,
                    inspection_time=end_time - start_time,
                )
            )

    # Spawn one process per job — all arrive at time 0
    for job in jobs:
        env.process(_inspect_job(job))

    env.run()

    # ── Aggregate metrics ────────────────────────────────────────────────────
    makespan = max((r.end_time for r in collected), default=0.0)
    wait_times = sorted(r.wait_time for r in collected)
    inspection_times = [r.inspection_time for r in collected]

    n = len(collected)
    mean_wait = sum(wait_times) / n if n > 0 else 0.0

    # p95 wait time (nearest-rank method)
    if n > 0:
        p95_idx = max(0, int(0.95 * n) - 1)
        p95_wait = wait_times[p95_idx]
    else:
        p95_wait = 0.0

    mean_insp = sum(inspection_times) / n if n > 0 else 0.0

    # Machine utilisation: total active work / (makespan × n_machines)
    total_work = sum(inspection_times)
    utilisation = (total_work / (makespan * n_machines)) if makespan > 0.0 else 0.0
    utilisation = min(utilisation, 1.0)  # cap at 1.0 due to float rounding

    throughput = (n / makespan) if makespan > 0.0 else 0.0

    return SimulationResult(
        schedule_name=schedule_name,
        n_jobs=len(jobs),
        n_machines=n_machines,
        makespan=round(makespan, 6),
        mean_wait_time=round(mean_wait, 6),
        p95_wait_time=round(p95_wait, 6),
        mean_inspection_time=round(mean_insp, 6),
        machine_utilisation=round(utilisation, 6),
        throughput=round(throughput, 6),
        jobs_completed=n,
        breakdown_count=breakdown_count_holder[0],
        job_results=collected,
    )


# ---------------------------------------------------------------------------
# High-level comparison entry point
# ---------------------------------------------------------------------------

def compare_des_schedules(
    segments: list[dict],
    x_lp: "np.ndarray",  # noqa: F821
    n_machines: int = 2,
    mttf: float = 0.0,
    mttr: float = 0.0,
    seed: int = 42,
) -> dict:
    """Simulate three queue orderings on IDENTICAL LP-allocated jobs.

    The LP allocation (x_lp) defines which segments to inspect and their
    inspection_times.  Three orderings of that SAME job set are simulated:

      lp_order      — sorted by LP fraction descending  (density proxy)
      naive_order   — sorted by raw anomaly score descending
      density_order — sorted by score/length density descending

    Because all three share identical inspection_times, makespan and machine
    utilisation are equal across orderings (same work, same capacity).  Only
    the wait-time distribution differs — isolating the pure effect of queue
    dispatch order from the allocation decision.

    Parameters
    ----------
    segments   : list of segment dicts (start, end, length, score)
    x_lp       : np.ndarray, shape (S,) — LP allocation fractions (defines job set)
    n_machines : int, default 2   — parallel inspection capacity
    mttf       : float, default 0.0 — mean time to failure (0 = no breakdowns)
    mttr       : float, default 0.0 — mean time to repair
    seed       : int, default 42   — RNG seed (identical for all three runs)

    Returns
    -------
    dict with keys:
        lp            — SimulationResult, LP-fraction ordering
        greedy        — SimulationResult, raw-score ordering  (naive)
        density       — SimulationResult, density ordering
        lp_wait_reduction_pct         — (naive_wait  − lp_wait)      / naive_wait  × 100
        density_wait_reduction_pct    — (naive_wait  − density_wait) / naive_wait  × 100
        lp_vs_density_wait_diff_pct   — (density_wait − lp_wait)     / density_wait × 100
                                        Near 0 confirms LP ≈ density ordering.
        lp_makespan_reduction_pct     — should be ~0 (same work)
        n_machines                    — int
        breakdown_enabled             — bool
    """
    # Build the job set from the LP allocation
    lp_jobs = schedule_from_allocation(segments, x_lp, schedule_name="lp_order")

    if not lp_jobs:
        empty = run_inspection_simulation([], n_machines, mttf, mttr, "empty", seed)
        return {
            "lp":                            empty,
            "greedy":                        empty,
            "density":                       empty,
            "lp_wait_reduction_pct":         0.0,
            "density_wait_reduction_pct":    0.0,
            "lp_vs_density_wait_diff_pct":   0.0,
            "lp_makespan_reduction_pct":     0.0,
            "n_machines":                    n_machines,
            "breakdown_enabled":             mttf > 0.0,
        }

    # ── Naive ordering: same jobs, sorted by raw score descending ──────────
    naive_jobs = sorted(lp_jobs, key=lambda j: -j.score)

    # ── Density ordering: same jobs, sorted by score/length density desc ───
    density_jobs = sorted(lp_jobs, key=lambda j: -(j.score / max(j.length, 1)))

    # ── Run all three simulations with IDENTICAL seed ───────────────────────
    lp_result      = run_inspection_simulation(lp_jobs,      n_machines, mttf, mttr, "lp_order",      seed)
    naive_result   = run_inspection_simulation(naive_jobs,   n_machines, mttf, mttr, "naive_order",   seed)
    density_result = run_inspection_simulation(density_jobs, n_machines, mttf, mttr, "density_order", seed)

    def _wait_reduction(baseline_wait, new_wait):
        if baseline_wait > 1e-9:
            return round(100.0 * (baseline_wait - new_wait) / baseline_wait, 2)
        return 0.0

    def _makespan_reduction(baseline, new):
        if baseline > 1e-9:
            return round(100.0 * (baseline - new) / baseline, 2)
        return 0.0

    naive_wait   = naive_result.mean_wait_time
    lp_wait      = lp_result.mean_wait_time
    density_wait = density_result.mean_wait_time

    return {
        "lp":      lp_result,
        "greedy":  naive_result,   # key kept as 'greedy' for backward compatibility
        "density": density_result,
        "lp_wait_reduction_pct":        _wait_reduction(naive_wait,   lp_wait),
        "density_wait_reduction_pct":   _wait_reduction(naive_wait,   density_wait),
        "lp_vs_density_wait_diff_pct":  _wait_reduction(density_wait, lp_wait),
        "lp_makespan_reduction_pct":    _makespan_reduction(naive_result.makespan, lp_result.makespan),
        "n_machines":                   n_machines,
        "breakdown_enabled":            mttf > 0.0,
    }
