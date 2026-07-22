# Dense convex-QP measured reference

This table is a **recorded reference, not a gate**. It captures the behavior of
the dense operator-splitting QP solver on a committed self-contained subset of
dense convex quadratic programs, one solve per problem with polish enabled. It
is updated deliberately — the same way benchmark baselines are — by rerunning
the runner and pasting its emitted table here; it is never asserted against in
CI. Accuracy *enforcement* lives in the parity test, where the solution is
checked against a trusted active-set oracle. This artifact exists to make the
solver's measured behavior on a broad, structurally varied reference set legible
and reviewable over time, under the project's ship-measure-report posture.

## What is measured

The subset stands in for the field-standard Maros–Mészáros dense-QP reference
set. Because that collection carries no confirmed redistribution license for
committing converted data, the committed problems here are **self-authored**
originals spanning the constraint structures the solver must handle (diagonal
and dense-SPD objectives; box-only, equality-only, mixed equality-plus-slack-box,
active general-inequality, and two-sided ranged rows). See
`tests/unit/mm_data/PROVENANCE.md` for the license decision and the path to
regenerate from a license-clean Maros–Mészáros mirror. Each problem's optimum
was computed by an independent KKT oracle, not by the solver under measurement.

Columns: solver status; iterations to termination; whether the polish step was
accepted; the unscaled infinity-norm primal and dual residuals; and the absolute
objective gap against the verified optimum.

## Recorded measurements

Source: `tests/unit/maros_meszaros_dense_test.cpp`. Regenerate by building
`argmin_unit_tests` (dev preset) and running the `[mm]` case, then replacing the
table below with its emitted block.

| problem | n | m | status | iters | polished | r_p | r_d | obj_gap |
|---|---|---|---|---|---|---|---|---|
| AGBOXA | 2 | 2 | solved | 50 | yes | 4.621e-22 | 8.882e-16 | 0.000e+00 |
| AGBOXB | 8 | 8 | solved | 50 | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| AGEQA | 3 | 1 | solved | 50 | yes | 0.000e+00 | 1.110e-16 | 0.000e+00 |
| AGEQB | 5 | 2 | solved | 50 | yes | 2.220e-16 | 8.882e-16 | 0.000e+00 |
| AGINEQA | 2 | 3 | solved | 25 | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| AGMIXA | 4 | 5 | solved | 50 | yes | 0.000e+00 | 2.220e-16 | 4.441e-16 |
| AGRNGA | 3 | 4 | solved | 50 | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |
