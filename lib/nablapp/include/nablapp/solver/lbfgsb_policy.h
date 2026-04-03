#ifndef HPP_GUARD_NABLAPP_SOLVER_LBFGSB_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_LBFGSB_POLICY_H

// L-BFGS-B solver policy for basic_solver.
//
// Implements the Limited-memory BFGS algorithm for Bound-constrained
// optimization. Each step computes: (1) Generalized Cauchy Point via
// breakpoint search along the projected gradient path, (2) subspace
// minimization over free variables using the compact L-BFGS reduced
// Hessian, (3) strong Wolfe line search capped at the maximum feasible
// step length.
//
// When the problem satisfies only differentiable (no bounds), all bounds
// default to +/-infinity and the algorithm reduces to standard L-BFGS
// with projected steps (D-04).
//
// Reference: Byrd, Lu, Nocedal, Zhu (1995) "A Limited Memory Algorithm
//            for Bound Constrained Optimization", SIAM J. Sci. Comput.
//            16(5), pp. 1190-1208.
//            N&W Sections 9.2 (compact representation), 16.6 (GCP +
//            subspace minimization).
//            K&W Section 6.5 (quasi-Newton methods).

#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/cauchy_point.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/detail/subspace_minimization.h"
#include "nablapp/line_search/strong_wolfe.h"
#include "nablapp/line_search/options.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace nablapp
{

template <int N = dynamic_dimension>
struct lbfgsb_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = lbfgsb_policy<M>;

    struct options_type
    {
        std::optional<std::uint8_t> history_depth{};  // L-BFGS memory depth (default: 10, N&W 7.2)
        line_search_options line_search{};             // Embedded line search params
    };

    options_type options{};

    // The eval_value and eval_gradient functors capture the Problem by reference.
    // The Problem object passed to init() MUST outlive the solver.
    // This is the same lifetime model as all nablapp policies (init takes
    // const Problem&).
    struct state_type
    {
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> g;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        double objective_value{};
        detail::compact_lbfgs<double, N> B;
        std::uint32_t iteration{0};

        std::function<double(const Eigen::Vector<double, N>&)> eval_value;
        std::function<void(const Eigen::Vector<double, N>&, Eigen::Vector<double, N>&)> eval_gradient;
    };

    template <typename Problem, typename Convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type s;

        s.x = x0;
        s.g.setZero(n);
        s.objective_value = problem.value(x0);
        problem.gradient(x0, s.g);

        if constexpr(bound_constrained<Problem>)
        {
            s.lower = problem.lower_bounds();
            s.upper = problem.upper_bounds();
        }
        else
        {
            constexpr double inf = std::numeric_limits<double>::infinity();
            s.lower = Eigen::Vector<double, N>::Constant(n, -inf);
            s.upper = Eigen::Vector<double, N>::Constant(n, inf);
        }

        // History depth: default 10 (N&W 7.2)
        std::uint8_t depth = options.history_depth.value_or(10);
        s.B = detail::compact_lbfgs<double, N>{depth};
        s.iteration = 0;

        s.eval_value = [&problem](const Eigen::Vector<double, N>& v) {
            return problem.value(v);
        };
        s.eval_gradient = [&problem](const Eigen::Vector<double, N>& v,
                                     Eigen::Vector<double, N>& grad) {
            problem.gradient(v, grad);
        };

        return s;
    }

    step_result<double> step(state_type& s)
    {
        // Evaluate gradient (skip on first iteration -- init already computed it)
        if(s.iteration != 0)
            s.eval_gradient(s.x, s.g);

        // Generalized Cauchy Point
        auto gcp = detail::cauchy_point(s.x, s.g, s.lower, s.upper, s.B);

        // Subspace minimization over free variables
        Eigen::Vector<double, N> x_new = detail::subspace_minimize(
            s.x, gcp.x_cauchy, s.g, s.lower, s.upper, gcp.free_indices, s.B);

        // Search direction
        Eigen::Vector<double, N> d = (x_new - s.x).eval();

        // Zero step check
        if(d.norm() < 1e-15)
        {
            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = s.g.norm(),
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
            };
        }

        // Maximum feasible step length (D-03)
        double alpha_max = detail::compute_alpha_max(s.x, d, s.lower, s.upper);

        // Fallback to Cauchy direction if alpha_max is too small
        if(alpha_max < 1e-15)
        {
            d = (gcp.x_cauchy - s.x).eval();
            if(d.norm() < 1e-15)
            {
                return step_result<double>{
                    .objective_value = s.objective_value,
                    .gradient_norm = s.g.norm(),
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                    .x_norm = s.x.norm(),
                };
            }
            alpha_max = detail::compute_alpha_max(s.x, d, s.lower, s.upper);
            if(alpha_max < 1e-15)
            {
                return step_result<double>{
                    .objective_value = s.objective_value,
                    .gradient_norm = s.g.norm(),
                    .step_size = 0.0,
                    .objective_change = 0.0,
                    .improved = false,
                    .x_norm = s.x.norm(),
                };
            }
        }

        // Line search functors (CORE-09: .eval() on Eigen expressions)
        auto phi = [&](double a) {
            return s.eval_value((s.x + a * d).eval());
        };
        auto dphi = [&](double a) {
            Eigen::Vector<double, N> g_temp(s.x.size());
            s.eval_gradient((s.x + a * d).eval(), g_temp);
            return g_temp.dot(d);
        };

        // Strong Wolfe with feasible step cap, using embedded line search options
        line_search_options ls_opts = options.line_search;
        ls_opts.max_alpha = std::min(ls_opts.max_alpha, alpha_max);
        auto ls = strong_wolfe(phi, dphi, s.objective_value, s.g.dot(d), ls_opts);

        // Update iterate (project for numerical safety)
        Eigen::Vector<double, N> x_old = s.x;
        double old_f = s.objective_value;
        s.x = detail::project((s.x + ls.alpha * d).eval(), s.lower, s.upper);

        // Evaluate new objective and gradient
        s.objective_value = s.eval_value(s.x);
        Eigen::Vector<double, N> new_g(s.x.size());
        s.eval_gradient(s.x, new_g);

        // Update L-BFGS curvature pairs
        Eigen::Vector<double, N> sk = (s.x - x_old).eval();
        Eigen::Vector<double, N> yk = (new_g - s.g).eval();
        s.B.push(sk, yk);

        // Store new gradient
        s.g = new_g;

        // Increment iteration
        ++s.iteration;

        double step_norm = sk.norm();
        double x_norm = s.x.norm();

        // Roundoff detection (N&W Section 3.4):
        // step too small relative to iterate magnitude
        std::optional<solver_status> policy_status{};
        constexpr double eps = std::numeric_limits<double>::epsilon();
        if(step_norm < eps * std::max(x_norm, 1.0) * 10.0)
            policy_status = solver_status::roundoff_limited;

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = new_g.norm(),
            .step_size = step_norm,
            .objective_change = s.objective_value - old_f,
            .improved = s.objective_value < old_f,
            .x_norm = x_norm,
            .policy_status = policy_status,
        };
    }

    // Hot start -- preserves curvature pairs (D-05).
    void reset(state_type& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = x0;
        s.objective_value = s.eval_value(x0);
        s.eval_gradient(x0, s.g);
        s.iteration = 0;
        // B (compact_lbfgs) is NOT reset -- preserves curvature pairs
    }

    // Cold restart -- clears curvature history (D-05).
    void reset_clear(state_type& s, const Eigen::Vector<double, N>& x0)
    {
        reset(s, x0);
        s.B.reset();
    }
};

}

#endif
