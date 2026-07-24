"""Cross-thread cancellation.

Every wait here is bounded from inside the test, because the pinned dependency
set carries no timeout plugin and an unbounded hang in continuous integration is
worse than a failed assertion.

These cases assert that a running solve is cancellable from another thread and
that the solver survives it. They deliberately do not claim to prove that the
binding's own release of the interpreter lock is what makes that possible: a
callback defined in the interpreter yields the lock on its own, both at bytecode
boundaries and inside any call that blocks, so cancellation lands whether or not
the binding released anything. Measured -- with the release removed and nothing
else changed, all of these still pass. The release earns its keep on the paths
these tests cannot reach: the solver's own work between callbacks, which runs no
bytecode and would otherwise hold the lock outright.
"""

import threading
import time

import numpy as np
import pytest

import argmin

DIMENSION = 20
CALLBACK_DELAY = 0.001
JOIN_TIMEOUT = 20.0
DEADLINE = 10.0


def rosenbrock(x):
    head = x[:-1]
    tail = x[1:]
    return float(np.sum((1.0 - head) ** 2 + 100.0 * (tail - head**2) ** 2))


def rosenbrock_gradient(x):
    g = np.zeros_like(x)
    head = x[:-1]
    tail = x[1:]
    g[:-1] += -2.0 * (1.0 - head) - 400.0 * head * (tail - head**2)
    g[1:] += 200.0 * (tail - head**2)
    return g


class SlowObjective:
    def __init__(self):
        self.calls = 0
        self.entered_loop = threading.Event()

    def arm(self):
        self.calls = 0
        self.entered_loop.clear()

    def __call__(self, x):
        self.calls += 1
        if self.calls >= 2:
            self.entered_loop.set()
        time.sleep(CALLBACK_DELAY)
        return rosenbrock(x)


def make_solver(objective):
    start = np.full(DIMENSION, -1.2)
    return argmin.LbfgsbSolver(
        objective, start, gradient=rosenbrock_gradient, max_iterations=1_000_000
    )


def run_in_worker(solver):
    outcome = {}

    def body():
        outcome["result"] = solver.solve()

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    return worker, outcome


def test_abort_from_another_thread_stops_a_running_solve():
    objective = SlowObjective()
    solver = make_solver(objective)
    objective.arm()

    started = time.monotonic()
    worker, outcome = run_in_worker(solver)

    assert objective.entered_loop.wait(timeout=DEADLINE), "the solve never entered its loop"
    solver.abort()

    worker.join(timeout=JOIN_TIMEOUT)
    elapsed = time.monotonic() - started
    assert not worker.is_alive(), f"the worker was still running after {JOIN_TIMEOUT} s"
    assert elapsed < DEADLINE, f"the abort took {elapsed:.3f} s"

    result = outcome["result"]
    assert result.status == argmin.SolverStatus.aborted
    assert result.iterations < 1000


def test_the_main_thread_keeps_running_while_a_solve_is_in_flight():
    objective = SlowObjective()
    solver = make_solver(objective)
    objective.arm()

    worker, outcome = run_in_worker(solver)
    assert objective.entered_loop.wait(timeout=DEADLINE)

    ticks = 0
    started = time.monotonic()
    while time.monotonic() - started < 0.05:
        ticks += 1
    assert ticks > 0

    solver.abort()
    worker.join(timeout=JOIN_TIMEOUT)
    assert not worker.is_alive()
    assert outcome["result"].status == argmin.SolverStatus.aborted


def test_an_abort_requested_before_the_loop_is_honored_immediately():
    objective = SlowObjective()
    solver = make_solver(objective)

    solver.abort()
    started = time.monotonic()
    result = solver.solve()
    elapsed = time.monotonic() - started

    assert result.status == argmin.SolverStatus.aborted
    assert result.iterations == 0
    assert elapsed < DEADLINE


def test_a_solver_is_reusable_after_an_abort():
    objective = SlowObjective()
    solver = make_solver(objective)
    objective.arm()

    worker, _ = run_in_worker(solver)
    assert objective.entered_loop.wait(timeout=DEADLINE)
    solver.abort()
    worker.join(timeout=JOIN_TIMEOUT)
    assert not worker.is_alive()

    solver.reset(np.full(DIMENSION, -1.2))
    resumed = solver.step_n(5)
    assert resumed.status != argmin.SolverStatus.aborted


def test_a_second_solve_on_a_busy_solver_is_rejected():
    objective = SlowObjective()
    solver = make_solver(objective)
    objective.arm()

    worker, _ = run_in_worker(solver)
    assert objective.entered_loop.wait(timeout=DEADLINE)

    with pytest.raises(argmin.ArgminError) as raised:
        solver.solve()
    assert raised.value.kind == argmin.ErrorKind.invalid_state

    solver.abort()
    worker.join(timeout=JOIN_TIMEOUT)
    assert not worker.is_alive()
