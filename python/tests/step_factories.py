"""The solver factories and registries test_step_control parametrizes over."""

import numpy as np

import argmin

PROBLEM = argmin.Rosenbrock(2)
START = PROBLEM.initial_point()
ELSEWHERE = np.array([0.5, -0.25])
BOX = (np.full(2, -5.0), np.full(2, 5.0))


def objective(x):
    return PROBLEM.value(x)


def gradient(x):
    return PROBLEM.gradient(x)


def inactive_constraint(x):
    return np.array([10.0])


def inactive_jacobian(x):
    return np.zeros((1, 2))


def make_lbfgsb():
    return argmin.LbfgsbSolver(
        objective, START, gradient=gradient, max_iterations=500
    )


def make_bobyqa():
    lower, upper = BOX
    return argmin.BobyqaSolver(
        objective, START, lower_bounds=lower, upper_bounds=upper, max_iterations=2000
    )


def make_slsqp():
    return argmin.SlsqpSolver(
        objective,
        START,
        gradient=gradient,
        constraints=inactive_constraint,
        constraint_jacobian=inactive_jacobian,
        num_equality=0,
        num_inequality=1,
        max_iterations=500,
    )


def make_filter_slsqp():
    return argmin.FilterSlsqpSolver(
        objective,
        START,
        gradient=gradient,
        constraints=inactive_constraint,
        constraint_jacobian=inactive_jacobian,
        num_equality=0,
        num_inequality=1,
        max_iterations=500,
    )


def make_tr_sqp():
    return argmin.TrSqpSolver(
        objective,
        START,
        gradient=gradient,
        constraints=inactive_constraint,
        constraint_jacobian=inactive_jacobian,
        num_equality=0,
        num_inequality=1,
        max_iterations=500,
    )


FACTORIES = {
    "lbfgsb": make_lbfgsb,
    "bobyqa": make_bobyqa,
    "slsqp": make_slsqp,
    "filter_slsqp": make_filter_slsqp,
    "tr_sqp": make_tr_sqp,
}

# BOBYQA rebuilds its whole interpolation model on either reset entry point, so
# the two are behaviorally indistinguishable for it; only the policies that
# carry curvature across a reset can show the difference.
STATE_CARRYING = ("lbfgsb", "slsqp", "filter_slsqp", "tr_sqp")
