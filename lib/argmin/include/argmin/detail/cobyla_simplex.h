#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_H

// Simplex driver engine for COBYLA (Powell 1994).
//
// COBYLA maintains a simplex of n+1 vertices in R^n. The last column of SIM
// holds the current best (optimal) vertex; the preceding n columns hold the
// displacements from the optimal vertex to the other vertices, and SIMI holds
// the inverse of that n-by-n displacement matrix. DATMAT stores, per vertex,
// the constraint values, the objective, and the greatest constraint violation.
//
// Each outer iteration builds linear approximations of the objective and every
// constraint by interpolation on the simplex, solves the two-stage linear
// trust-region subproblem (TRSTLP), evaluates the trial point, adapts the
// merit penalty PARMU toward feasibility only when reaching feasibility costs
// objective, and either revises the simplex or maintains its geometry. The
// trust radius RHO is reduced from RHOBEG to RHOEND, and termination occurs
// once RHO reaches RHOEND after a short, geometrically acceptable step.
//
// The merit function is  phi(x) = f(x) + parmu * RESMAX(x)  where RESMAX is the
// MAXIMUM constraint violation (all constraints are written in the >= 0 sense).
//
// This engine is a faithful port of M. J. D. Powell's COBYLA2 (the cobylb
// routine), restructured as a resumable state machine: init() bootstraps the
// simplex with n+1 evaluations, and each advance() performs one further
// function evaluation (a geometry-improving vertex or a trust-region trial)
// together with the surrounding bookkeeping, until RHO reaches RHOEND.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization method that
//            models the objective and constraint functions by linear
//            interpolation." Sections 2-5.

#include "argmin/detail/cobyla_engine.h"
#include "argmin/detail/cobyla_bootstrap.h"
#include "argmin/detail/cobyla_trial_step.h"
#include "argmin/detail/cobyla_engine_cycle.h"
#include "argmin/detail/cobyla_geometry_step.h"
#include "argmin/detail/cobyla_vertex_models.h"
#include "argmin/detail/cobyla_simplex_update.h"

#endif
