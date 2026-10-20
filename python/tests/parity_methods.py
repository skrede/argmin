"""The wrapped solvers and method registries test_native_parity parametrizes over."""

import numpy as np

import argmin
import argmin.experimental as experimental

BUDGET = 500
# The library's default gradient threshold stops the trust-region method one
# iteration short of the objective tolerance the parity tests assert; this one
# moves no other constrained method's stopping iteration.
CONSTRAINED_SOLVE = dict(max_iterations=BUDGET, gradient_threshold=1e-6)
BOX = (np.full(2, -5.0), np.full(2, 5.0))
EXPERIMENTAL_NATIVE = ("bobyqa", "cmaes")


def native_solve(problem, method, **keywords):
    if method in EXPERIMENTAL_NATIVE:
        return experimental.solve_native(problem, method, **keywords)
    return problem.solve_native(method, **keywords)


def rosenbrock_problem():
    lower, upper = BOX
    return argmin.Rosenbrock(2, lower_bounds=lower, upper_bounds=upper)


def reexpressed_rosenbrock(x):
    return float((1.0 - x[0]) ** 2 + 5.0 * (x[1] - x[0] ** 2) ** 2)


def reexpressed_rosenbrock_gradient(x):
    return np.array(
        [
            -2.0 * (1.0 - x[0]) - 20.0 * x[0] * (x[1] - x[0] ** 2),
            10.0 * (x[1] - x[0] ** 2),
        ]
    )


def reexpressed_constrained(x):
    return float((1.0 - x[0]) ** 2)


def reexpressed_constrained_gradient(x):
    return np.array([-2.0 * (1.0 - x[0]), 0.0])


def reexpressed_constraints(x):
    return np.array([10.0 * (x[1] - x[0] ** 2)])


def reexpressed_constraint_jacobian(x):
    return np.array([[-20.0 * x[0], 10.0]])


def wrapped_lbfgsb(problem, objective, gradient):
    lower, upper = BOX
    return argmin.LbfgsbSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        lower_bounds=lower,
        upper_bounds=upper,
        max_iterations=BUDGET,
    )


def wrapped_bobyqa(problem, objective, gradient):
    lower, upper = BOX
    return experimental.BobyqaSolver(
        objective,
        problem.initial_point(),
        lower_bounds=lower,
        upper_bounds=upper,
        max_iterations=BUDGET,
    )


def wrapped_kraft_slsqp(problem, objective, gradient, constraints, jacobian):
    return argmin.KraftSlsqpSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        constraints=constraints,
        constraint_jacobian=jacobian,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        **CONSTRAINED_SOLVE,
    )


def wrapped_filter_slsqp(problem, objective, gradient, constraints, jacobian):
    return argmin.FilterSlsqpSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        constraints=constraints,
        constraint_jacobian=jacobian,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        **CONSTRAINED_SOLVE,
    )


def wrapped_tr_sqp(problem, objective, gradient, constraints, jacobian):
    return argmin.TrSqpSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        constraints=constraints,
        constraint_jacobian=jacobian,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        **CONSTRAINED_SOLVE,
    )


def wrapped_cobyla(problem, objective, gradient, constraints, jacobian):
    return argmin.CobylaSolver(
        objective,
        problem.initial_point(),
        constraints=constraints,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        **CONSTRAINED_SOLVE,
    )


UNCONSTRAINED_METHODS = {"lbfgsb": wrapped_lbfgsb, "bobyqa": wrapped_bobyqa}
CONSTRAINED_METHODS = {
    "kraft_slsqp": wrapped_kraft_slsqp,
    "filter_slsqp": wrapped_filter_slsqp,
    "tr_sqp": wrapped_tr_sqp,
    "cobyla": wrapped_cobyla,
}
