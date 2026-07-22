"""Each solver against a closed-form optimum, and outcomes that are answers.

The reference problem is

    minimize  0.5 x^T P x + q^T x   subject to   x0 + x1 = 1,  0 <= xi <= 0.7

with P = [[4, 1], [1, 2]] and q = [1, 1]. Substituting x1 = 1 - x0 reduces the
objective to 2 x0^2 - x0 + 2, whose unconstrained minimizer 0.25 is cut off by
x1 <= 0.7, so the optimum is x = (0.3, 0.7) with objective 1.88. The expected
values below are that closed form, not another solver's output.
"""

import numpy as np
import pytest
import scipy.sparse as sp

OPTIMUM = np.array([0.3, 0.7])
OPTIMAL_OBJECTIVE = 1.88
FEASIBLE_START = np.array([0.5, 0.5])


def reference_problem():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    return P, q, A, l, u


def solve_with(argmin, kind, P, q, A, l, u, options=None):
    args = () if options is None else (options,)
    if kind == "sparse_admm":
        solver = argmin.SparseAdmmQpSolver()
        return solver, solver.solve(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u, *args)
    solver = getattr(argmin, "DenseAdmmQpSolver" if kind == "dense_admm" else "DenseActiveSetQpSolver")(
        P.shape[0], A.shape[0]
    )
    if kind == "dense_active_set":
        solver.warm_start(FEASIBLE_START)
    return solver, solver.solve(P, q, A, l, u, *args)


def capped_options(argmin, kind):
    options = argmin.SparseQpOptions() if kind == "sparse_admm" else argmin.DenseQpOptions()
    options.max_iterations = 1
    options.check_termination = 1
    options.adaptive_rho = False
    options.polish = False
    return options


ALL = ["dense_admm", "dense_active_set", "sparse_admm"]
ADMM = ["dense_admm", "sparse_admm"]


@pytest.mark.parametrize("kind", ALL)
def test_the_analytic_optimum_is_reached(argmin, kind):
    _, result = solve_with(argmin, kind, *reference_problem())
    assert result.status == argmin.QpStatus.solved
    assert np.allclose(result.x, OPTIMUM, atol=1e-6)
    assert result.objective_value == pytest.approx(OPTIMAL_OBJECTIVE, abs=1e-6)


@pytest.mark.parametrize("kind", ALL)
def test_the_iteration_cap_is_an_attribute_and_not_an_exception(argmin, kind):
    _, result = solve_with(argmin, kind, *reference_problem(), options=capped_options(argmin, kind))
    assert result.status == argmin.QpStatus.max_iterations
    assert result.x.shape == (2,)


@pytest.mark.parametrize("kind", ADMM)
def test_infeasibility_is_an_attribute_and_not_an_exception(argmin, kind):
    P, q, A, _, _ = reference_problem()
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.2, 0.2])
    _, result = solve_with(argmin, kind, P, q, A, l, u)
    assert result.status == argmin.QpStatus.primal_infeasible


# The active-set adapter documents a feasible-start contract: it cannot certify
# infeasibility, it rejects the start it was given. That is an argument
# violation and belongs on the error channel, not in the status attribute.
def test_the_active_set_solver_rejects_a_start_it_cannot_use(argmin):
    P, q, A, _, _ = reference_problem()
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.2, 0.2])
    solver = argmin.DenseActiveSetQpSolver(2, 3)
    solver.warm_start(np.array([0.1, 0.1]))
    with pytest.raises(argmin.ArgminError) as raised:
        solver.solve(P, q, A, l, u)
    assert raised.value.kind == argmin.ErrorKind.infeasible_start


@pytest.mark.parametrize("kind", ALL)
def test_resolve_agrees_with_a_fresh_solve_on_the_updated_vectors(argmin, kind):
    P, q, A, l, u = reference_problem()
    updated_q = np.array([-3.0, 0.0])

    solver, first = solve_with(argmin, kind, P, q, A, l, u)
    assert np.allclose(first.x, OPTIMUM, atol=1e-6)
    resolved = solver.resolve(updated_q, l, u)

    _, fresh = solve_with(argmin, kind, P, updated_q, A, l, u)
    assert fresh.status == argmin.QpStatus.solved
    assert resolved.status == argmin.QpStatus.solved
    assert not np.allclose(fresh.x, OPTIMUM, atol=1e-3), "the update must move the optimum"
    assert np.allclose(resolved.x, fresh.x, atol=1e-6)
