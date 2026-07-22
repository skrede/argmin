import re
import gc

import numpy as np

NUMERIC_FIELD = re.compile(r"(\w+)=([-+0-9.eE]+)")


def qp_result(argmin):
    return argmin.QpResult(
        x=np.array([1.5, -2.25, 3.0]),
        y=np.array([0.125, 0.25]),
        status=argmin.QpStatus.solved_inaccurate,
        iterations=17,
        polished=True,
        primal_residual=1.25e-09,
        dual_residual=2.5e-10,
        objective_value=-1.0 / 3.0,
    )


def solve_result(argmin):
    return argmin.SolveResult(
        status=argmin.SolverStatus.converged,
        iterations=9,
        function_evaluations=31,
        objective_value=-1.0 / 3.0,
        gradient_norm=1.5e-08,
        constraint_violation=0.25,
        x=np.array([1.5, -2.25]),
    )


def test_qp_result_fields_are_readable(argmin):
    result = qp_result(argmin)

    assert result.status == argmin.QpStatus.solved_inaccurate
    assert result.iterations == 17
    assert result.polished is True
    assert result.primal_residual == 1.25e-09
    assert result.dual_residual == 2.5e-10
    assert result.objective_value == -1.0 / 3.0
    assert result.x.tolist() == [1.5, -2.25, 3.0]
    assert result.y.tolist() == [0.125, 0.25]


def test_solve_result_fields_are_readable(argmin):
    result = solve_result(argmin)

    assert result.status == argmin.SolverStatus.converged
    assert result.iterations == 9
    assert result.function_evaluations == 31
    assert result.objective_value == -1.0 / 3.0
    assert result.gradient_norm == 1.5e-08
    assert result.constraint_violation == 0.25
    assert result.x.tolist() == [1.5, -2.25]


def test_qp_result_arrays_outlive_the_producing_object(argmin):
    x, y = (lambda result: (result.x, result.y))(qp_result(argmin))
    gc.collect()

    assert x.dtype == np.float64 and y.dtype == np.float64
    assert x.flags.owndata and y.flags.owndata
    assert x.tolist() == [1.5, -2.25, 3.0]
    assert y.tolist() == [0.125, 0.25]


def test_solve_result_array_outlives_the_producing_object(argmin):
    x = (lambda result: result.x)(solve_result(argmin))
    gc.collect()

    assert x.dtype == np.float64
    assert x.flags.owndata
    assert x.tolist() == [1.5, -2.25]


def test_reprs_round_trip_their_numbers(argmin):
    for result in (qp_result(argmin), solve_result(argmin)):
        text = repr(result)
        fields = dict(NUMERIC_FIELD.findall(text))

        assert fields
        for name, rendered in fields.items():
            assert float(rendered) == getattr(result, name)
        assert fields["objective_value"] == repr(result.objective_value)
