"""Measured speedup across four workers, which is what proves the lock is released.

A failure here means one of two things: the interpreter lock is not released
across the solve, or something inside the released region reacquires it. Either
way independent solves serialize, and a workload split four ways takes about as
long as the same workload run on one thread.

The assertion is therefore a SPEEDUP bound, not a slowdown bound. With the lock
held, four workers doing the identical total work land at a ratio near one; a
"four workers under 1.5x single-threaded" form passes cleanly in that state at
any work quantum above roughly a hundred microseconds, and the per-solve time
here is milliseconds, so that form would ship a dropped release green. Correct
behavior on four cores lands near three to five tenths.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytestmark = pytest.mark.slow

WORKERS = 4
SOLVES_PER_WORKER = 8
TOTAL_SOLVES = WORKERS * SOLVES_PER_WORKER
SPEEDUP_RATIO = 0.7
MINIMUM_SOLVE_SECONDS = 5e-4


def usable_cores() -> int:
    for probe in ("process_cpu_count", "sched_getaffinity"):
        query = getattr(os, probe, None)
        if query is None:
            continue
        count = query() if probe == "process_cpu_count" else len(query(0))
        if count:
            return count
    return os.cpu_count() or 1


def timed_problem(size: int = 100):
    rng = np.random.default_rng(20260722)
    rows = 2 * size
    root = rng.standard_normal((size, size))
    P = root @ root.T + size * np.eye(size)
    q = rng.standard_normal(size)
    A = rng.standard_normal((rows, size))
    return P, q, A, -np.ones(rows), np.ones(rows)


@pytest.mark.skipif(
    usable_cores() < WORKERS,
    reason=f"a speedup assertion needs {WORKERS} usable cores, this host offers {usable_cores()}",
)
def test_a_fixed_workload_split_four_ways_is_measurably_faster(argmin):
    P, q, A, l, u = timed_problem()
    rows, size = A.shape

    # One solver per worker: these solvers are stateful, and sharing one across
    # threads would be a data race rather than the defect under test.
    solvers = [argmin.DenseAdmmQpSolver(size, rows) for _ in range(WORKERS)]
    for solver in solvers:
        assert solver.solve(P, q, A, l, u).status == argmin.QpStatus.solved

    start = time.perf_counter()
    for _ in range(TOTAL_SOLVES):
        solvers[0].solve(P, q, A, l, u)
    single_threaded = time.perf_counter() - start

    per_solve = single_threaded / TOTAL_SOLVES
    assert per_solve > MINIMUM_SOLVE_SECONDS, (
        f"a single solve takes {per_solve * 1e6:.0f} us, which is too close to the "
        "timing noise floor for this measurement to mean anything; raise the problem size"
    )

    def run_share(worker: int) -> None:
        solver = solvers[worker]
        for _ in range(SOLVES_PER_WORKER):
            solver.solve(P, q, A, l, u)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(run_share, range(WORKERS)))
    across_workers = time.perf_counter() - start

    ratio = across_workers / single_threaded
    assert ratio <= SPEEDUP_RATIO, (
        f"the same {TOTAL_SOLVES} solves took {single_threaded * 1e3:.1f} ms on one thread "
        f"and {across_workers * 1e3:.1f} ms across {WORKERS} workers (ratio {ratio:.2f}, "
        f"required at most {SPEEDUP_RATIO}); a ratio near one means the interpreter lock "
        "was held across the solve or reacquired inside the released region"
    )


@pytest.mark.skipif(
    usable_cores() < WORKERS,
    reason=f"a speedup assertion needs {WORKERS} usable cores, this host offers {usable_cores()}",
)
def test_every_worker_reaches_the_same_answer(argmin):
    P, q, A, l, u = timed_problem(40)
    rows, size = A.shape

    def solve_once(_: int) -> np.ndarray:
        solver = argmin.DenseAdmmQpSolver(size, rows)
        result = solver.solve(P, q, A, l, u)
        assert result.status == argmin.QpStatus.solved
        return result.x

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = list(pool.map(solve_once, range(4 * WORKERS)))

    for answer in answers[1:]:
        assert np.allclose(answer, answers[0], atol=1e-8)
