# Sparse convex-QP measured reference

This table is a **recorded reference, not a gate**. It captures the behavior of
the sparse operator-splitting QP solver on the committed problem set and on the
structured control-shaped family, one solve per problem with polish enabled. It
is updated deliberately — the same way benchmark baselines are — by rerunning
the runner and pasting its emitted table here; it is never asserted against in
CI. Accuracy *enforcement* lives elsewhere: in the dense–sparse parity leg,
where the sparse answer is checked against the verified dense solver and against
an independent active-set oracle, and in the control-shaped leg, where every
accepted iterate is graded by a first-order optimality check computed from the
original unscaled data. This artifact exists to make the solver's measured
behavior legible and reviewable over time, under the project's
ship-measure-report posture.

## What is measured

Two groups share one table.

The **committed** group is the same small self-authored dense convex-QP set the
dense path is measured against, loaded into column-major sparse form with a zero
drop tolerance so the sparse solver sees exactly the same problem. Each of these
carries an optimum computed by an independent KKT oracle, not by the solver
under measurement; see `tests/unit/mm_data/PROVENANCE.md` for the license
decision behind that set and for the path to regenerate from a license-clean
mirror. These problems exercise no sparsity and no control structure — at this
size the density column is a large fraction — which is precisely why the second
group exists.

The **control-shaped** group is generated in-tree by
`tests/unit/sparse_control_qp_family.h`: a spring-coupled double-integrator
chain over a runtime horizon, with banded equality dynamics, box-bounded inputs
and slack-softened state bounds. It is the workload this solver exists for, at
sizes where sparsity is load-bearing rather than decorative. It is generated
rather than committed as fixture data because no redistribution-license-clean
source of structured sparse reference problems is available to this repository.

Columns: the problem group; the label; the decision-vector length `n`; the row
count `m`; the stored-entry count of the constraint matrix and its density
against `m · n`; the solver status; iterations to termination; whether the
polish step was accepted; the unscaled infinity-norm primal and dual residuals;
the objective value; and the absolute objective gap against the verified
optimum. **The objective-gap column is empty for the control-shaped group**: the
generated family has no independently verified optimum, and filling that column
from the solver's own answer would be a fabricated number rather than a
measurement.

## Recorded measurements

Source: `tests/unit/sparse_qp_reference_test.cpp`. Regenerate by building
`argmin_unit_tests` (dev preset) and running the `[sparse_ref]` case, then
replacing the table below with its emitted block:

```
cmake --build build/dev --target argmin_unit_tests
./build/dev/tests/unit/argmin_unit_tests "sparse QP ship-measure-report"
```

| group | problem | n | m | nnz(A) | density | status | iters | polished | r_p | r_d | obj | obj_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| committed | AGBOXA | 2 | 2 | 2 | 5.000e-01 | solved | 50 | yes | 4.621e-22 | 0.000e+00 | -8.000000e+00 | 0.000e+00 |
| committed | AGBOXB | 8 | 8 | 8 | 1.250e-01 | solved | 50 | yes | 0.000e+00 | 0.000e+00 | -4.196652e+00 | 0.000e+00 |
| committed | AGEQA | 3 | 1 | 3 | 1.000e+00 | solved | 50 | yes | 0.000e+00 | 2.220e-16 | -2.000000e+00 | 0.000e+00 |
| committed | AGEQB | 5 | 2 | 7 | 7.000e-01 | solved | 50 | yes | 4.441e-16 | 8.882e-16 | 2.870139e+00 | 1.776e-15 |
| committed | AGINEQA | 2 | 3 | 4 | 6.667e-01 | solved | 25 | yes | 0.000e+00 | 0.000e+00 | -1.937500e+00 | 0.000e+00 |
| committed | AGMIXA | 4 | 5 | 8 | 4.000e-01 | solved | 50 | yes | 1.110e-16 | 2.220e-16 | -8.687500e-01 | 0.000e+00 |
| committed | AGRNGA | 3 | 4 | 6 | 5.000e-01 | solved | 50 | yes | 0.000e+00 | 0.000e+00 | 5.000000e-01 | 0.000e+00 |
| control-shaped | H=10 | 156 | 276 | 616 | 1.431e-02 | solved | 75 | yes | 1.026e-16 | 7.994e-15 | 3.078012e+01 |   |
| control-shaped | H=30 | 456 | 816 | 1836 | 4.934e-03 | solved | 200 | yes | 1.138e-15 | 5.107e-15 | -9.524834e+01 |   |
| control-shaped | H=60 | 906 | 1626 | 3666 | 2.489e-03 | solved | 125 | yes | 1.343e-15 | 7.105e-15 | -3.057015e+01 |   |

## What this document does not say

The runner's only hard assertions are honesty properties: the call returned a
value, the reported status is one of the terminal outcomes, and every reported
quantity is finite. It asserts no accuracy bound, no iteration bound and no
density bound. Nothing here is a timing measurement, and nothing here is an
allocation measurement — this solver allocates, reads no clock, and makes no
real-time claim. Its allocation posture is published in
`docs/rt-safety-matrix.md`.
