"""The ownership edge the library documents in prose.

The policy stores a raw pointer to the problem and requires it to outlive the
solver. Nothing on the interpreter side honors prose, so every case here drops
each caller-side reference, forces a collection, and only then uses the solver.
"""

import gc
import weakref

import numpy as np
import pytest

import argmin

START = np.array([-1.2, 1.0])
OPTIMUM = np.array([1.0, 1.0])


def rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_gradient(x):
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )


def assert_solved(result):
    assert result.status == argmin.SolverStatus.converged
    assert np.allclose(result.x, OPTIMUM, atol=1e-4)


def test_a_plain_function_survives_losing_every_caller_reference():
    def objective(x):
        return rosenbrock(x)

    def gradient(x):
        return rosenbrock_gradient(x)

    solver = argmin.LbfgsbSolver(objective, START, gradient=gradient, max_iterations=400)
    del objective, gradient
    gc.collect()

    assert_solved(solver.solve())


def test_a_bound_method_survives_losing_its_object():
    class Problem:
        def __init__(self, scale):
            self.scale = scale

        def value(self, x):
            return self.scale * rosenbrock(x)

        def gradient(self, x):
            return self.scale * rosenbrock_gradient(x)

    owner = Problem(1.0)
    solver = argmin.LbfgsbSolver(
        owner.value, START, gradient=owner.gradient, max_iterations=400
    )
    tracker = weakref.ref(owner)
    del owner
    gc.collect()

    assert tracker() is not None
    assert_solved(solver.solve())


def make_closure_objective(seen):
    # The captured array is built here and never named in the caller, so once
    # the returned closure is dropped the adapter's reference is the only one
    # left keeping the array alive.
    captured = np.linspace(0.5, 1.5, 4096)

    def objective(x):
        seen["sum"] = float(captured.sum())
        return rosenbrock(x)

    return objective, float(captured.sum())


def test_a_closure_keeps_its_captured_array_intact():
    seen = {}
    objective, expected_sum = make_closure_objective(seen)

    def gradient(x):
        return rosenbrock_gradient(x)

    solver = argmin.LbfgsbSolver(objective, START, gradient=gradient, max_iterations=400)
    del objective, gradient
    gc.collect()

    assert_solved(solver.solve())
    assert seen["sum"] == pytest.approx(expected_sum)


def test_a_result_outlives_the_solver_that_produced_it():
    solver = argmin.LbfgsbSolver(
        rosenbrock, START, gradient=rosenbrock_gradient, max_iterations=400
    )
    result = solver.solve()
    kept = result.x.copy()

    del solver
    gc.collect()

    assert np.array_equal(result.x, kept)
    assert np.allclose(result.x, OPTIMUM, atol=1e-4)


def test_many_solvers_are_constructed_and_destroyed_without_a_crash():
    for _ in range(200):
        solver = argmin.LbfgsbSolver(
            rosenbrock, START, gradient=rosenbrock_gradient, max_iterations=50
        )
        solver.step()
        del solver
        gc.collect()


def test_the_iterate_is_a_copy_the_caller_owns():
    solver = argmin.LbfgsbSolver(
        rosenbrock, START, gradient=rosenbrock_gradient, max_iterations=400
    )
    first = solver.x
    first[:] = 42.0
    assert np.allclose(solver.x, START)


def test_a_solver_in_a_reference_cycle_is_collectable():
    class Holder:
        def __init__(self):
            self.solver = None

        def value(self, x):
            return rosenbrock(x)

        def gradient(self, x):
            return rosenbrock_gradient(x)

    holder = Holder()
    holder.solver = argmin.LbfgsbSolver(
        holder.value, START, gradient=holder.gradient, max_iterations=400
    )
    assert_solved(holder.solver.solve())
    tracker = weakref.ref(holder)

    del holder
    gc.collect()

    assert tracker() is None
