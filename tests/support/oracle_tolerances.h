#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_TOLERANCES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_ORACLE_TOLERANCES_H

#include "oracle/qp_oracle.h"
#include "oracle/kkt_measure.h"

#include "argmin/solver/options.h"
#include "argmin/solver/convergence.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/options/convergence_defaults.h"
#include "argmin/options/dense_active_set_qp_options.h"

#include <tuple>
#include <type_traits>

namespace argmin::oracle
{

template <typename Criterion, typename Criteria>
struct tuple_holds;

template <typename Criterion, typename... Criteria>
struct tuple_holds<Criterion, std::tuple<Criteria...>> : std::bool_constant<(std::is_same_v<Criterion, Criteria> || ...)>
{
};

// std::get<T> on a tuple without T is a hard error inside the standard library, not a substitution
// failure, so presence is decided from the tuple's element types.
template <typename Criterion, typename Convergence>
concept has_criterion = tuple_holds<Criterion, decltype(Convergence::criteria)>::value;

template <typename Criterion, typename Convergence, typename Scalar>
Scalar stationarity_threshold_or(const Convergence& convergence, Scalar fallback)
{
    if constexpr(has_criterion<Criterion, Convergence>)
        return std::get<Criterion>(convergence.criteria).stationarity_threshold;
    else
        return fallback;
}

// ftol_reached fires only when the KKT measure is below the objective criterion's stationarity
// threshold, so that threshold is the claim's stationarity tolerance.
template <typename Convergence, typename Scalar = typename Convergence::scalar_type>
Scalar objective_stationarity_or(const Convergence& convergence, Scalar fallback)
{
    const Scalar relative = stationarity_threshold_or<objective_tolerance_rel_criterion<Scalar>>(convergence, fallback);
    return stationarity_threshold_or<objective_tolerance_criterion<Scalar>>(convergence, relative);
}

template <typename Convergence, typename Scalar = typename Convergence::scalar_type>
Scalar converged_stationarity(const Convergence& convergence)
{
    if constexpr(has_criterion<gradient_tolerance_criterion<Scalar>, Convergence>)
        return std::get<gradient_tolerance_criterion<Scalar>>(convergence.criteria).threshold;
    else
        return objective_stationarity_or(convergence, convergence_defaults<Scalar>::gradient_threshold);
}

template <typename Convergence, typename Scalar = typename Convergence::scalar_type>
tolerances<Scalar> tolerances_of(const solver_options<Convergence>& opts)
{
    const Scalar feasibility = opts.constraint_tolerance.value_or(opts.feasibility_tolerance);
    const Scalar ftol = objective_stationarity_or(opts.convergence, convergence_defaults<Scalar>::stationarity_threshold);
    return tolerances<Scalar>(converged_stationarity(opts.convergence), feasibility, ftol);
}

template <typename Scalar>
qp_tolerances<Scalar> qp_tolerances_of(const dense_qp_options<Scalar>& opts)
{
    return qp_tolerances<Scalar>(opts.eps_abs, opts.eps_rel, opts.eps_prim_inf, opts.eps_dual_inf);
}

template <typename Scalar>
qp_tolerances<Scalar> qp_tolerances_of(const sparse_qp_options<Scalar>& opts)
{
    return qp_tolerances<Scalar>(opts.eps_abs, opts.eps_rel, opts.eps_prim_inf, opts.eps_dual_inf);
}

// The active-set method has a single tolerance, which its terminal KKT check applies relative to the gradient and
// multiplier scale, so it is both the absolute and the relative term here.
template <typename Scalar>
qp_tolerances<Scalar> qp_tolerances_of(const dense_active_set_qp_options<Scalar>& opts)
{
    return qp_tolerances<Scalar>(opts.tolerance, opts.tolerance, opts.tolerance, opts.tolerance);
}

}

#endif
