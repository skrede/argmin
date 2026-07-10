#ifndef HPP_GUARD_ARGMIN_DETAIL_LBFGSB_DIRECTION_H
#define HPP_GUARD_ARGMIN_DETAIL_LBFGSB_DIRECTION_H

// Shared direction computation for L-BFGS-B policies.
//
// Provides a unified compute_direction helper that selects the optimal
// direction computation path:
//   1. Compile-time unconstrained: two-loop recursion d = -H*g (no bounds).
//   2. Runtime all-free: two-loop recursion when all variables are interior.
//   3. GCP + subspace minimization: full L-BFGS-B direction with active bounds.
//
// The fast paths (1, 2) produce the same direction as standard L-BFGS but
// skip the O(n log n) breakpoint sort and O(nf^3) reduced Hessian factorization
// of the GCP+subspace path when no bounds are active.
//
// Reference: N&W Algorithm 9.1 (two-loop recursion),
//            N&W Section 16.6 (GCP + subspace minimization),
//            Byrd, Lu, Nocedal, Zhu (1995) "A Limited Memory Algorithm
//            for Bound Constrained Optimization".

#include "argmin/detail/cauchy_point.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/detail/subspace_minimization.h"
#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <optional>

namespace argmin::detail
{

// Result of direction computation: search direction and maximum feasible step.

template <typename Scalar, int N>
struct direction_result
{
    Eigen::Vector<Scalar, N> d;
    Scalar alpha_max;
};

// Check whether all variables are free (no bound is active given current x and g).
//
// A variable i has an active bound if it sits at a bound and the steepest
// descent direction -g pushes into that bound:
//   - x[i] at lower AND g[i] > 0 (descent pushes toward lower)
//   - x[i] at upper AND g[i] < 0 (descent pushes toward upper)
//
// Returns false on first active variable found (early exit).
// Only checks finite bounds (matching cauchy_point.h breakpoint generation).
//
// Reference: Byrd et al. 1995 (breakpoint definition, active set).

template <typename Scalar, int N>
bool all_variables_free(const Eigen::Vector<Scalar, N>& x,
                        const Eigen::Vector<Scalar, N>& g,
                        const Eigen::Vector<Scalar, N>& lower,
                        const Eigen::Vector<Scalar, N>& upper)
{
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
    constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();
    const int n = static_cast<int>(x.size());

    for(int i = 0; i < n; ++i)
    {
        if(lower[i] > -inf &&
           x[i] <= lower[i] + eps * (Scalar(1) + std::abs(lower[i])) &&
           g[i] > Scalar(0))
            return false;

        if(upper[i] < inf &&
           x[i] >= upper[i] - eps * (Scalar(1) + std::abs(upper[i])) &&
           g[i] < Scalar(0))
            return false;
    }

    return true;
}

// Compute search direction using the optimal path for the current bound state.
//
// Returns nullopt when no viable direction exists (zero step).
//
// Branch 1 (compile-time unconstrained): d = -H*g via two-loop recursion,
//   alpha_max = infinity. No bound projection or alpha_max computation.
//   Reference: N&W Algorithm 9.1.
//
// Branch 2 (runtime all-free): d = -H*g via two-loop recursion,
//   alpha_max = compute_alpha_max(x, d, lower, upper). Falls through to
//   Branch 3 if alpha_max is degenerate (near-zero).
//   Reference: N&W Algorithm 9.1, Byrd et al. 1995.
//
// Branch 3 (GCP + subspace): full L-BFGS-B direction computation with
//   Cauchy fallback on degenerate alpha_max.
//   Reference: N&W Section 16.6.

template <typename Problem, typename Scalar = double, int N = argmin::dynamic_dimension>
std::optional<direction_result<Scalar, N>> compute_direction(
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& g,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const auto& B,
    cauchy_point_solver<Scalar, N>& gcp_solver,
    subspace_minimizer<Scalar, N>& ssm_solver)
{
    // Branch 1: compile-time unconstrained -- pure two-loop recursion.
    // N&W Algorithm 9.1: d = -H_k * g.
    if constexpr(!bound_constrained<Problem>)
    {
        Eigen::Vector<Scalar, N> d = -(B.two_loop_recursion(g)).eval();
        if(d.norm() < 1e-15)
            return std::nullopt;
        return direction_result<Scalar, N>{
            .d = std::move(d),
            .alpha_max = std::numeric_limits<Scalar>::infinity(),
        };
    }
    else
    {
        // Branch 2: runtime all-free -- two-loop recursion with alpha_max.
        // Byrd et al. 1995: skip GCP breakpoint search when no bound is active.
        if(all_variables_free<Scalar, N>(x, g, lower, upper))
        {
            Eigen::Vector<Scalar, N> d = -(B.two_loop_recursion(g)).eval();
            if(d.norm() < 1e-15)
                return std::nullopt;

            Scalar alpha_max = compute_alpha_max<Scalar, N>(x, d, lower, upper);

            // Degenerate geometry: fall through to GCP path.
            if(alpha_max >= 1e-15)
            {
                return direction_result<Scalar, N>{
                    .d = std::move(d),
                    .alpha_max = alpha_max,
                };
            }
        }

        // Branch 3: GCP + subspace minimization (N&W Section 16.6).
        const auto& gcp = gcp_solver.solve(x, g, lower, upper, B);

        Eigen::Vector<Scalar, N> x_new = ssm_solver.solve(
            x, gcp.x_cauchy, g, lower, upper, gcp.free_indices, B);

        Eigen::Vector<Scalar, N> d = (x_new - x).eval();

        if(d.norm() < 1e-15)
            return std::nullopt;

        Scalar alpha_max = compute_alpha_max<Scalar, N>(x, d, lower, upper);

        // Fallback to Cauchy direction if alpha_max is too small.
        if(alpha_max < 1e-15)
        {
            d = (gcp.x_cauchy - x).eval();
            if(d.norm() < 1e-15)
                return std::nullopt;

            alpha_max = compute_alpha_max<Scalar, N>(x, d, lower, upper);
            if(alpha_max < 1e-15)
                return std::nullopt;
        }

        return direction_result<Scalar, N>{
            .d = std::move(d),
            .alpha_max = alpha_max,
        };
    }
}

}

#endif
