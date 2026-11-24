"""A failed L-BFGS-B line search holds the iterate and ends the solve with
line_search_failed, or non_finite_evaluation when the search met a non-finite
value, rather than committing the step."""

import numpy as np
import pytest

import argmin

SUCCESS_STATUSES = frozenset(
    {
        argmin.SolverStatus.converged,
        argmin.SolverStatus.xtol_reached,
        argmin.SolverStatus.ftol_reached,
    }
)

POISONS = [
    pytest.param(float("inf"), argmin.SolverStatus.non_finite_evaluation, id="infinite"),
    pytest.param(1.0e6, argmin.SolverStatus.line_search_failed, id="finite-cliff"),
]


class PoisonedQuadratic:
    """Once armed, the poison everywhere except at the anchor; the boundary rejects
    NaN but passes infinity. The explicit gradient stays finite, since a
    finite-difference gradient would read the poisoned values."""

    def __init__(self, poison):
        self.armed = False
        self.anchor = None
        self.poison = poison

    def value(self, x):
        if self.armed and not np.array_equal(x, self.anchor):
            return self.poison
        return float((x[0] - 1.0) ** 2 + 10.0 * (x[1] + 0.5) ** 2)

    @staticmethod
    def gradient(x):
        return np.array([2.0 * (x[0] - 1.0), 20.0 * (x[1] + 0.5)])


def armed_solver(poison):
    problem = PoisonedQuadratic(poison)
    solver = argmin.LbfgsbSolver(
        problem.value, np.array([3.0, 2.0]), gradient=problem.gradient, max_iterations=100
    )
    solver.step()
    solver.step()
    problem.anchor = np.array(solver.x)
    problem.armed = True
    return solver, problem


@pytest.mark.parametrize("poison, status", POISONS)
def test_a_failed_line_search_holds_the_iterate(poison, status):
    solver, problem = armed_solver(poison)
    solver.step()
    assert np.array_equal(solver.x, problem.anchor)
    assert solver.status() == status

    result = solver.solve()
    assert result.status == status
    assert result.status not in SUCCESS_STATUSES


@pytest.mark.parametrize("poison, status", POISONS)
def test_a_failed_line_search_ends_the_solve_without_success(poison, status):
    solver, problem = armed_solver(poison)
    result = solver.solve()
    assert result.status == status
    assert result.status not in SUCCESS_STATUSES
    assert np.array_equal(result.x, problem.anchor)
