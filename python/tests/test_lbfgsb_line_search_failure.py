"""A failed L-BFGS-B line search holds the iterate and ends the solve with
line_search_failed rather than committing the step."""

import numpy as np

import argmin

SUCCESS_STATUSES = frozenset(
    {
        argmin.SolverStatus.converged,
        argmin.SolverStatus.xtol_reached,
        argmin.SolverStatus.ftol_reached,
    }
)


class PoisonedQuadratic:
    """Once armed, infinite everywhere except at the anchor; the boundary rejects
    NaN but passes infinity. The explicit gradient stays finite, since a
    finite-difference gradient would read the poisoned values."""

    def __init__(self):
        self.armed = False
        self.anchor = None

    def value(self, x):
        if self.armed and not np.array_equal(x, self.anchor):
            return float("inf")
        return float((x[0] - 1.0) ** 2 + 10.0 * (x[1] + 0.5) ** 2)

    @staticmethod
    def gradient(x):
        return np.array([2.0 * (x[0] - 1.0), 20.0 * (x[1] + 0.5)])


def armed_solver():
    problem = PoisonedQuadratic()
    solver = argmin.LbfgsbSolver(
        problem.value, np.array([3.0, 2.0]), gradient=problem.gradient, max_iterations=100
    )
    solver.step()
    solver.step()
    problem.anchor = np.array(solver.x)
    problem.armed = True
    return solver, problem


def test_a_failed_line_search_holds_the_iterate():
    solver, problem = armed_solver()
    solver.step()
    assert np.array_equal(solver.x, problem.anchor)
    assert solver.status() == argmin.SolverStatus.line_search_failed

    result = solver.solve()
    assert result.status == argmin.SolverStatus.line_search_failed
    assert result.status not in SUCCESS_STATUSES


def test_a_failed_line_search_ends_the_solve_without_success():
    solver, problem = armed_solver()
    result = solver.solve()
    assert result.status == argmin.SolverStatus.line_search_failed
    assert result.status not in SUCCESS_STATUSES
    assert np.array_equal(result.x, problem.anchor)
