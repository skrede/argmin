#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_POLICY_BRIDGE_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_POLICY_BRIDGE_H

#include "argmin/detail/lagrangian.h"
#include "argmin/detail/bound_projection.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <cstdint>
#include <optional>
#include <algorithm>

namespace argmin::detail
{

// The automatic radius, 10% of the largest finite bound range, is capped at half the smallest
// one, Powell's rule that every initial vertex fits in the box: Powell (2009), "The BOBYQA
// algorithm for bound constrained optimization without derivatives", DAMTP 2009/NA06.
template <typename Scalar>
inline Scalar cobyla_auto_trust_radius(Scalar initial_trust_radius, const Eigen::VectorX<Scalar>& lower,
                                       const Eigen::VectorX<Scalar>& upper, std::int32_t n)
{
    if(initial_trust_radius > Scalar(0.0))
        return initial_trust_radius;
    Scalar max_range = Scalar(0.0);
    Scalar min_range = std::numeric_limits<Scalar>::infinity();
    for(int i = 0; i < n; ++i)
    {
        Scalar range_i = upper[i] - lower[i];
        if(std::isfinite(range_i))
        {
            max_range = std::max(max_range, range_i);
            min_range = std::min(min_range, range_i);
        }
    }
    if(!(max_range > Scalar(0.0)))
        return Scalar(1.0);
    return std::min(Scalar(0.1) * max_range, Scalar(0.5) * min_range);
}

template <typename Scalar>
inline std::int32_t cobyla_first_invalid_coordinate(const Eigen::VectorX<Scalar>& lower,
                                                    const Eigen::VectorX<Scalar>& upper)
{
    for(Eigen::Index i = 0; i < lower.size(); ++i)
    {
        if(!(lower[i] < upper[i]))
            return static_cast<std::int32_t>(i);
    }
    return -1;
}

// A rejected state keeps x0 as given, since an empty box has no projection, and never builds
// an engine; its objective is NaN because nothing is ever evaluated.
template <typename State, typename Vector>
inline bool cobyla_reject_invalid_box(State& s, const Vector& x0)
{
    using Scalar = typename decltype(s.x)::Scalar;
    s.invalid_coordinate = cobyla_first_invalid_coordinate(s.lower, s.upper);
    s.invalid_problem = s.invalid_coordinate >= 0;
    if(s.invalid_problem)
    {
        s.x = x0;
        s.objective_value = std::numeric_limits<Scalar>::quiet_NaN();
        s.iteration = 0;
    }
    return s.invalid_problem;
}

template <typename Scalar, typename State>
inline step_result<Scalar> cobyla_invalid_step_report(const State& s)
{
    return step_result<Scalar>{
        .objective_value = s.objective_value,
        .gradient_norm = Scalar(0.0),
        .step_size = Scalar(0.0),
        .objective_change = Scalar(0.0),
        .improved = false,
        .is_null_step = true,
        .constraint_violation = Scalar(0.0),
        .policy_status = solver_status::invalid_problem,
        .evaluations = 0,
    };
}

// Copy the engine incumbent into the policy state and refresh the split
// constraint vectors used for violation reporting.
template <typename State>
inline void cobyla_sync_from_engine(State& s)
{
    using Scalar = typename decltype(s.x)::Scalar;
    s.x = detail::project(s.engine->x(), s.lower, s.upper);
    s.objective_value = s.engine->objective_value();
    s.parmu = s.engine->parmu();
    s.clamped_evaluations = s.engine->clamped_evaluations();

    const int m_total = s.n_eq + s.n_ineq;
    if(m_total > 0)
    {
        Eigen::VectorX<Scalar> c_full(m_total);
        s.problem->constraints(s.x, c_full);
        s.c_eq = c_full.head(s.n_eq);
        s.c_ineq = c_full.tail(s.n_ineq);
    }
    else
    {
        s.c_eq.resize(0);
        s.c_ineq.resize(0);
    }
}

// COBYLA owns its own termination: it iterates until the trust radius
// RHO contracts to RHOEND and then reports xtol_reached via
// policy_status. Until then the reported progress magnitudes are kept
// at the trust-radius scale so step_budget_solver's step / objective / stall
// criteria do not declare convergence prematurely -- the derivative-
// free incumbent (the simplex pole) can hold still for several
// evaluations while the model and geometry improve. The gradient proxy
// is likewise floored while RHO has not reached RHOEND.
template <typename State, typename Scalar>
inline step_result<Scalar> cobyla_step_report(const State& s, bool running, bool improved, Scalar final_trust_radius)
{
    std::optional<solver_status> policy_status{};
    if(!running)
        policy_status = s.engine->status();

    const Scalar rho = s.engine->rho();
    Scalar grad_proxy = s.engine->gradient_proxy();
    if(running && rho > final_trust_radius)
        grad_proxy = std::max(grad_proxy, Scalar(1.0));

    return step_result<Scalar>{
        .objective_value = s.objective_value,
        .gradient_norm = grad_proxy,
        .step_size = running ? rho : Scalar(0.0),
        .objective_change = running ? rho : Scalar(0.0),
        .improved = improved,
        .is_null_step = !running,
        .constraint_violation = detail::constraint_violation(s.c_eq, s.c_ineq),
        .policy_status = policy_status,
    };
}

}

#endif
