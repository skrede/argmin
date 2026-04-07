#ifndef HPP_GUARD_NABLAPP_SOLVER_BOBYQA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_BOBYQA_POLICY_H

// BOBYQA solver policy for basic_solver (Powell 2009 faithful implementation).
//
// Implements Powell's Bound Optimization BY Quadratic Approximation using
// the GOPT/HQ/PQ model representation with BMAT/ZMAT H-matrix factorization.
// The main iteration uses TRSBOX for trust-region steps, ALTMOV for geometry
// improvement, and RESCUE for interpolation system recovery.
//
// This replaces the earlier SVD-based quadratic model approach with Powell's
// O(n^2) incremental update scheme, enabling proper geometry maintenance
// and denominator-based point replacement.
//
// Requires: objective<P,S> && bound_constrained<P,S>.
// No gradient is needed -- BOBYQA uses only objective evaluations.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06.
//            K&W Section 8.4 (surrogate model framework).

#include "nablapp/detail/bobyqa_model.h"
#include "nablapp/detail/bobyqa_update.h"
#include "nablapp/detail/bobyqa_trsbox.h"
#include "nablapp/detail/bobyqa_altmov.h"
#include "nablapp/detail/bobyqa_rescue.h"
#include "nablapp/detail/bound_projection.h"
#include "nablapp/result/step_result.h"
#include "nablapp/solver/options.h"
#include "nablapp/types.h"

#include "nablapp/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace nablapp
{

template <int N = dynamic_dimension>
struct bobyqa_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = bobyqa_policy<M>;

    struct options_type
    {
        std::optional<std::uint16_t> num_interpolation_points{};
        std::optional<double> initial_trust_radius{};
        std::optional<double> final_trust_radius{};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        double objective_value{};
        detail::bobyqa_model<double, N> model;
        double delta{};                            // Trust-region radius (varies within [rho, 2*rho])
        std::uint32_t iteration{0};
        std::uint16_t itest{0};
        std::uint16_t nf_since_rho_change{0};      // Steps since last rho contraction
        bool initialized{false};
    };

    template <typename Problem, typename Convergence>
        requires objective<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    const solver_options<Convergence>& opts, const options_type& policy_opts)
    {
        options = policy_opts;
        return init(problem, x0, opts);
    }

    template <typename Problem, typename Convergence = default_convergence>
        requires objective<Problem> && bound_constrained<Problem>
    state_type<Problem> init(const Problem& problem,
                    const Eigen::Vector<double, N>& x0,
                    [[maybe_unused]] const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        // Initial trust-region radius (Powell 2009).
        //
        // Auto heuristic: use the minimum finite bound range scaled by 0.2,
        // capped to avoid absurd initial steps for loosely bounded problems.
        // When bounds are very wide (> 100), fall back to a scale based on
        // ||x0|| or 1.0, whichever is larger. This matches the behavior of
        // NLopt's BOBYQA default radius selection.
        double rhobeg = options.initial_trust_radius.value_or(0.0);
        if(rhobeg <= 0.0)
        {
            double min_finite_range = std::numeric_limits<double>::infinity();
            for(int i = 0; i < n; ++i)
            {
                double range_i = s.upper[i] - s.lower[i];
                if(std::isfinite(range_i) && range_i > 0.0)
                    min_finite_range = std::min(min_finite_range, range_i);
            }

            if(std::isfinite(min_finite_range) && min_finite_range <= 100.0)
            {
                rhobeg = 0.2 * min_finite_range;
            }
            else
            {
                // Wide or unbounded: use x0 scale or 1.0.
                double x_scale = x0.norm();
                rhobeg = std::max(x_scale * 0.1, 1.0);
            }
        }

        double rhoend = options.final_trust_radius.value_or(1e-8);

        // Project x0 to feasible region.
        Eigen::Vector<double, N> x0_proj = detail::project(x0, s.lower, s.upper);

        // Initialize the Powell 2009 model (PRELIM).
        auto eval_fn = [&](const Eigen::Vector<double, N>& pt) -> double {
            return problem.value(pt);
        };

        s.model.initialize(x0_proj, s.lower, s.upper, rhobeg, rhoend, eval_fn);

        s.x = s.model.x_base + s.model.x_opt;
        s.objective_value = s.model.f_opt;
        s.delta = rhobeg;
        s.initialized = true;
        s.iteration = 0;
        s.itest = 0;
        s.nf_since_rho_change = 0;

        return s;
    }

    // Main iteration step per Powell 2009.
    //
    // Uses separate trust-region radius delta that varies based on
    // accuracy ratio. Delta floats within [rho, 2*rho]. When delta = rho
    // and steps are insufficient, rho is contracted.
    //
    // Reference: Powell 2009, main iteration loop.
    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.model.gopt.size();
        const double eps = std::numeric_limits<double>::epsilon();
        double old_f = s.objective_value;

        // Step 1: Compute TRSBOX step with current delta.
        Eigen::Vector<double, N> d = detail::trsbox(s.model, s.lower, s.upper, s.delta);

        double d_norm = d.norm();

        // Step 2: Short-step check.
        //
        // If ||d|| < 0.5 * rho, the model cannot make progress at this scale.
        // Contract rho or declare convergence.
        // Reference: Powell 2009, Sec. 5.
        if(d_norm < 0.5 * s.model.rho)
        {
            if(s.model.rho > s.model.rho_end)
            {
                s.model.update_rho();
                s.delta = s.model.rho;
                s.nf_since_rho_change = 0;
                ++s.iteration;
                return make_continuing_result(s, old_f);
            }

            return step_result<double>{
                .objective_value = s.objective_value,
                .gradient_norm = 0.0,
                .step_size = 0.0,
                .objective_change = 0.0,
                .improved = false,
                .x_norm = s.x.norm(),
            };
        }

        // Step 3: Evaluate trial point.
        Eigen::Vector<double, N> x_trial = detail::project(
            (s.model.x_base + s.model.x_opt + d).eval(), s.lower, s.upper);
        double f_new = s.problem->value(x_trial);
        ++s.nf_since_rho_change;

        // Accuracy ratio (Powell 2009, Sec. 5).
        double predicted = -s.model.evaluate(d);
        double actual = old_f - f_new;
        double ratio = (std::abs(predicted) > eps * 100.0)
                           ? actual / predicted
                           : 0.0;

        // Step 4: Update trust-region radius delta based on ratio.
        //
        // Good ratio (> 0.7) and step near boundary: expand delta.
        // Poor ratio (< 0.1): contract delta toward rho.
        // Reference: Powell 2009, Sec. 5.
        if(ratio >= 0.7 && d_norm >= 0.5 * s.delta)
        {
            s.delta = std::min(2.0 * s.delta, 2.0 * s.model.rho);
        }
        else if(ratio < 0.1)
        {
            s.delta = std::max(0.5 * s.delta, s.model.rho);
        }

        // Step 5: Select replacement and update model.
        auto vlag = detail::compute_vlag(s.model, d);
        double beta = detail::compute_beta(s.model, d);
        std::uint16_t knew = detail::select_replacement_powell(s.model, vlag, beta, f_new);

        double hk = detail::h_diagonal(s.model.zmat, static_cast<int>(knew));
        double denom = detail::compute_denominator(beta, hk, vlag[knew]);

        if(std::abs(denom) < eps * std::max(vlag[knew] * vlag[knew], 1.0))
        {
            Eigen::Vector<double, N> d_alt = detail::altmov(
                s.model, knew, s.lower, s.upper, s.delta);

            if(d_alt.norm() > eps * s.model.rho)
            {
                auto vlag_alt = detail::compute_vlag(s.model, d_alt);
                double beta_alt = detail::compute_beta(s.model, d_alt);
                double hk_alt = detail::h_diagonal(s.model.zmat, static_cast<int>(knew));
                double denom_alt = detail::compute_denominator(beta_alt, hk_alt, vlag_alt[knew]);

                if(std::abs(denom_alt) > std::abs(denom))
                {
                    d = d_alt;
                    x_trial = detail::project(
                        (s.model.x_base + s.model.x_opt + d).eval(), s.lower, s.upper);
                    f_new = s.problem->value(x_trial);
                    vlag = vlag_alt;
                    beta = beta_alt;
                    denom = denom_alt;
                }
            }

            if(std::abs(denom) < eps * std::max(vlag[knew] * vlag[knew], 1.0))
            {
                auto rescue_eval = [&](const Eigen::Vector<double, N>& pt) -> double {
                    return s.problem->value(pt);
                };
                detail::rescue(s.model, s.lower, s.upper, rescue_eval);
                s.delta = s.model.rho;

                s.x = s.model.x_base + s.model.x_opt;
                s.objective_value = s.model.f_opt;
                s.itest = 0;
                ++s.iteration;
                return make_continuing_result(s, old_f);
            }
        }

        // Update model with the new point.
        detail::update_model_after_replacement(s.model, knew, d, f_new, vlag, beta);

        // Track improvement.
        bool improved = false;
        if(f_new < s.objective_value)
        {
            s.objective_value = f_new;
            s.x = s.model.x_base + s.model.x_opt;
            improved = true;
            s.itest = 0;
        }
        else
        {
            s.x = s.model.x_base + s.model.x_opt;
        }

        // Model switching heuristic (BOB-04).
        if(ratio <= 0.0)
            ++s.itest;
        else
            s.itest = 0;

        if(s.itest >= 3)
        {
            absorb_hq_into_pq(s.model, n);
            s.itest = 0;
        }

        // Rho contraction trigger: if many steps without improvement and
        // delta is already at rho, force rho contraction.
        // Reference: Powell 2009, Sec. 5.
        const int npt = s.model.pq.size();
        if(!improved && s.nf_since_rho_change > static_cast<std::uint16_t>(2 * npt)
           && s.model.rho > s.model.rho_end)
        {
            s.model.update_rho();
            s.delta = s.model.rho;
            s.nf_since_rho_change = 0;
        }

        // Base shift when x_opt grows large.
        if(s.model.x_opt.norm() > 1e3 * s.model.rho)
            s.model.shift_base(n);

        ++s.iteration;

        double obj_change = s.objective_value - old_f;
        double effective_step = improved ? d_norm : s.delta;
        double effective_change = improved ? obj_change : s.delta;

        double grad_proxy = s.model.gopt.norm();
        if(s.model.rho > s.model.rho_end)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = effective_change,
            .improved = improved,
            .x_norm = s.x.norm(),
        };
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        Eigen::Vector<double, N> x0_proj = detail::project(x0, s.lower, s.upper);

        auto eval_fn = [&](const Eigen::Vector<double, N>& pt) -> double {
            return s.problem->value(pt);
        };

        s.model.initialize(x0_proj, s.lower, s.upper,
                           s.model.rho_beg, s.model.rho_end, eval_fn);

        s.x = s.model.x_base + s.model.x_opt;
        s.objective_value = s.model.f_opt;
        s.delta = s.model.rho;
        s.iteration = 0;
        s.itest = 0;
        s.nf_since_rho_change = 0;
        s.initialized = true;
    }

private:
    // Absorb HQ (explicit Hessian) into PQ (implicit Lagrange weights).
    //
    // This is the model switching step: zero out HQ and adjust PQ so
    // that the model value and gradient remain unchanged.
    //
    // For each PQ[k], add the contribution from HQ via:
    //   PQ[k] += xpt[k]^T * HQ_mat * xpt[k] / (xpt[k]^T xpt[k])^2
    //
    // This is approximate but sufficient for the model switch heuristic.
    //
    // Reference: Powell 2009 (model switching between full and min-Frobenius).
    template <typename Scalar, int NN, int NPT>
    static void absorb_hq_into_pq(detail::bobyqa_model<Scalar, NN, NPT>& model, int n)
    {
        const int npt = model.pq.size();

        for(int k = 0; k < npt; ++k)
        {
            Scalar xpt_sq = model.xpt.row(k).squaredNorm();
            if(xpt_sq < std::numeric_limits<Scalar>::epsilon())
                continue;

            // Compute xpt[k]^T HQ xpt[k] using packed storage.
            Scalar quad = Scalar(0);
            for(int i = 0; i < n; ++i)
            {
                for(int j = 0; j < n; ++j)
                    quad += model.hq_element(i, j, n) * model.xpt(k, i) * model.xpt(k, j);
            }

            model.pq[k] += quad / (xpt_sq * xpt_sq) * xpt_sq;
        }

        model.hq.setZero();
    }

    // Build a step_result for iterations that loop back (rho contraction, rescue).
    template <typename P>
    static step_result<double> make_continuing_result(const state_type<P>& s, double old_f)
    {
        double grad_proxy = s.model.gopt.norm();
        if(s.model.rho > s.model.rho_end)
            grad_proxy = std::max(grad_proxy, 1.0);

        return step_result<double>{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = s.model.rho,
            .objective_change = s.model.rho,
            .improved = false,
            .x_norm = s.x.norm(),
        };
    }
};

}

#endif
