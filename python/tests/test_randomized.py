"""Seeded randomized sweeps over the conversion and callback space.

The per-surface modules sample the conversion space at a few chosen points.
This one explores it: dimension, row count, dtype, memory order, contiguity,
writability, conditioning, bound tightness and degenerate rows, several hundred
generated cases in all.

This is seeded randomization with a recorded seed, NOT shrinking-based property
testing. The pinned dependency set carries no property-testing framework, so
there is no shrinker and no minimal counterexample: a failure reports the master
seed and the case index, and that pair reproduces the case exactly. Nobody
should read the word "randomized" here as the stronger thing.

The invariant is a disjunction with exactly two arms and no third outcome
permitted: solving the varied form either agrees with the canonical form or
raises the typed exception. A varied form that returns a *different* answer
without raising is a failure, and so is a crash. Both arms are asserted to have
actually been taken over the sweep, because a disjunction only one arm of which
is ever exercised proves half of what it claims.

On the dtype arm specifically: a narrower dtype genuinely loses information, so
the canonical form for that comparison is the same data round-tripped through
the narrower dtype and back. That isolates the question this module is asking --
does the boundary convert the way the array library does -- from the unrelated
question of how sensitive a given quadratic program is to a perturbation of its
data.
"""

import numpy as np
import pytest

import argmin

MASTER_SEED = 20260722

QP_CASES = 60
NLP_CASES = 40

BUDGET = 500


def rng_for(case):
    return np.random.default_rng([MASTER_SEED, case])


def where(case, variant):
    return f"master seed {MASTER_SEED}, case {case}, variant {variant!r}"


def spd_matrix(rng, n, ill_conditioned):
    basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
    spread = np.logspace(0.0, -6.0 if ill_conditioned else -1.0, n)
    matrix = basis @ np.diag(spread) @ basis.T
    return 0.5 * (matrix + matrix.T)


def generate_problem(case):
    rng = rng_for(case)
    n = int(rng.integers(2, 7))
    m = int(rng.integers(0, 7))

    P = spd_matrix(rng, n, ill_conditioned=bool(case % 3 == 0))
    q = rng.standard_normal(n)
    A = rng.standard_normal((m, n))

    if m >= 2 and case % 4 == 0:
        A[m - 1] = A[0]
    if m >= 3 and case % 5 == 0:
        A[m - 2] = 0.0

    center = A @ rng.standard_normal(n) if m else np.zeros(0)
    width = np.abs(rng.standard_normal(m))
    if case % 6 == 0:
        width[:] = 0.0
    l = center - width
    u = center + width
    if m >= 2 and case % 7 == 0:
        l[0] = -np.inf
        u[1] = np.inf

    return P, q, A, l, u


def contiguous_double(arrays):
    return tuple(np.ascontiguousarray(a, dtype=np.float64) for a in arrays)


def as_fortran(arrays):
    return tuple(np.asfortranarray(a) if a.ndim == 2 else a.copy() for a in arrays)


def as_strided_slice(arrays):
    varied = []
    for a in arrays:
        if a.ndim == 2:
            padded = np.zeros((a.shape[0], a.shape[1] * 2))
            padded[:, ::2] = a
            varied.append(padded[:, ::2])
        else:
            padded = np.zeros(a.size * 2)
            padded[::2] = a
            varied.append(padded[::2])
    return tuple(varied)


def as_read_only(arrays):
    varied = []
    for a in arrays:
        copy = a.copy()
        copy.setflags(write=False)
        varied.append(copy)
    return tuple(varied)


def as_single_precision(arrays):
    return tuple(a.astype(np.float32) for a in arrays)


def truncate_the_linear_term(arrays):
    P, q, A, l, u = arrays
    return P, q[:-1].copy(), A, l, u


def poison_the_linear_term(arrays):
    P, q, A, l, u = arrays
    q = q.copy()
    q[0] = np.nan
    return P, q, A, l, u


# The value-preserving variants must agree exactly; the dtype variant is
# compared against a canonical form demoted the same way; the last two must
# raise. Nothing else is permitted for any of them.
VARIANTS = {
    "fortran_order": (as_fortran, None),
    "strided_slice": (as_strided_slice, None),
    "read_only": (as_read_only, None),
    "single_precision": (as_single_precision, lambda arrays: as_single_precision(arrays)),
    "truncated": (truncate_the_linear_term, None),
    "not_a_number": (poison_the_linear_term, None),
}


def solve_dense(arrays):
    P, q, A, l, u = arrays
    solver = argmin.DenseAdmmQpSolver(P.shape[0], A.shape[0])
    return solver.solve(P, q, A, l, u)


def agrees(first, second):
    return (
        first.status == second.status
        and first.iterations == second.iterations
        and first.objective_value == second.objective_value
        and np.array_equal(first.x, second.x)
        and np.array_equal(first.y, second.y)
    )


def test_a_varied_form_either_agrees_or_raises(capsys):
    agreed = 0
    raised = 0

    for case in range(QP_CASES):
        arrays = generate_problem(case)
        for variant, (vary, demote) in VARIANTS.items():
            reference = contiguous_double(demote(arrays) if demote else arrays)
            canonical = solve_dense(reference)
            varied = vary(arrays)

            try:
                answer = solve_dense(varied)
            except argmin.ArgminError:
                raised += 1
                continue

            assert agrees(canonical, answer), where(case, variant)
            agreed += 1

    with capsys.disabled():
        print(
            f"\nrandomized conversion sweep: {agreed} agreed, {raised} raised, "
            f"{QP_CASES * len(VARIANTS)} cases at master seed {MASTER_SEED}"
        )

    # Both arms of the disjunction have to be reachable for the invariant to
    # mean anything.
    assert agreed > 0 and raised > 0
    assert agreed + raised == QP_CASES * len(VARIANTS)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_variant_family_takes_one_arm_consistently(variant):
    vary, demote = VARIANTS[variant]
    outcomes = set()

    for case in range(QP_CASES):
        arrays = generate_problem(case)
        canonical = solve_dense(contiguous_double(demote(arrays) if demote else arrays))
        try:
            outcomes.add(agrees(canonical, solve_dense(vary(arrays))))
        except argmin.ArgminError:
            outcomes.add("raised")

    assert outcomes in ({True}, {"raised"}), (variant, outcomes, MASTER_SEED)


# The agreement arm of the invariant is only worth as much as the predicate
# behind it, and a predicate that cannot return False proves nothing.
def test_the_agreement_predicate_discriminates():
    first = solve_dense(contiguous_double(generate_problem(0)))
    assert agrees(first, first)
    for case in range(1, QP_CASES):
        other = solve_dense(contiguous_double(generate_problem(case)))
        if other.x.shape == first.x.shape:
            assert not agrees(first, other), case
            return
    pytest.fail("no second case of the same shape to compare against")


def test_the_generated_problems_span_the_intended_space():
    rows = set()
    lengths = set()
    infinite = 0
    tight = 0
    for case in range(QP_CASES):
        _, q, A, l, u = generate_problem(case)
        lengths.add(q.size)
        rows.add(A.shape[0])
        infinite += int(np.isinf(l).any() or np.isinf(u).any())
        tight += int(A.shape[0] > 0 and np.array_equal(l, u))

    assert lengths >= {2, 3, 4, 5, 6}
    assert 0 in rows and max(rows) >= 5
    assert infinite > 0
    assert tight > 0


def native_case(case):
    rng = rng_for(1000 + case)
    n = int(rng.integers(2, 6))
    lower = np.full(n, -5.0)
    upper = np.full(n, 5.0)
    problem = argmin.Rosenbrock(n, lower_bounds=lower, upper_bounds=upper)
    x0 = rng.uniform(-2.0, 2.0, size=n)
    return problem, x0, lower, upper


# The interpreter-defined objective here delegates straight back to the
# compiled evaluator, so the arithmetic is identical on both paths and the only
# difference is the round trip. Anything but exact agreement is the boundary
# corrupting a value.
def test_a_randomized_interpreter_objective_matches_the_compiled_one():
    for case in range(NLP_CASES):
        problem, x0, lower, upper = native_case(case)

        native = problem.solve_native("lbfgsb", x0=x0, max_iterations=BUDGET)
        interpreted = argmin.LbfgsbSolver(
            problem.value,
            x0,
            gradient=problem.gradient,
            lower_bounds=lower,
            upper_bounds=upper,
            max_iterations=BUDGET,
        ).solve()

        assert native.status == interpreted.status, where(case, "native_parity")
        assert native.iterations == interpreted.iterations, where(case, "native_parity")
        assert native.objective_value == interpreted.objective_value, where(
            case, "native_parity"
        )
        assert np.array_equal(native.x, interpreted.x), where(case, "native_parity")


def test_a_randomized_starting_point_of_the_wrong_length_raises(capsys):
    raised = 0
    for case in range(NLP_CASES):
        problem, x0, lower, upper = native_case(case)
        with pytest.raises(argmin.ArgminError) as caught:
            argmin.LbfgsbSolver(
                problem.value,
                np.concatenate([x0, [0.0]]),
                lower_bounds=lower,
                upper_bounds=upper,
            )
        assert caught.value.kind == argmin.ErrorKind.dimension_mismatch, where(case, "long_x0")
        raised += 1

    with capsys.disabled():
        print(f"randomized nonlinear sweep: {raised} clean rejections at master seed {MASTER_SEED}")
    assert raised == NLP_CASES


def test_the_seed_is_recorded_and_reproduces_a_case():
    first = generate_problem(7)
    again = generate_problem(7)
    other = generate_problem(8)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(
        a.shape == b.shape and np.array_equal(a, b) for a, b in zip(first, other)
    )
