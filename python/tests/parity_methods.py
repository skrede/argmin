"""The wrapped solvers and method registries test_native_parity parametrizes over."""

import numpy as np

import argmin

BUDGET = 500
BOX = (np.full(2, -5.0), np.full(2, 5.0))


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
    return argmin.BobyqaSolver(
        objective,
        problem.initial_point(),
        lower_bounds=lower,
        upper_bounds=upper,
        max_iterations=BUDGET,
    )


def wrapped_slsqp(problem, objective, gradient, constraints, jacobian):
    return argmin.SlsqpSolver(
        objective,
        problem.initial_point(),
        gradient=gradient,
        constraints=constraints,
        constraint_jacobian=jacobian,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        max_iterations=BUDGET,
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
        max_iterations=BUDGET,
    )


def wrapped_cobyla(problem, objective, gradient, constraints, jacobian):
    return argmin.CobylaSolver(
        objective,
        problem.initial_point(),
        constraints=constraints,
        num_equality=problem.num_equality(),
        num_inequality=problem.num_inequality(),
        max_iterations=BUDGET,
    )


UNCONSTRAINED_METHODS = {"lbfgsb": wrapped_lbfgsb, "bobyqa": wrapped_bobyqa}
CONSTRAINED_METHODS = {
    "slsqp": wrapped_slsqp,
    "filter_slsqp": wrapped_filter_slsqp,
    "cobyla": wrapped_cobyla,
}
