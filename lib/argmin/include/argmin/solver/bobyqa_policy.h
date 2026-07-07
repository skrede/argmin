#ifndef HPP_GUARD_ARGMIN_SOLVER_BOBYQA_POLICY_H
#define HPP_GUARD_ARGMIN_SOLVER_BOBYQA_POLICY_H

// BOBYQA solver policy for basic_solver.
//
// Implements Powell's Bound Optimization BY Quadratic Approximation.
// A trust-region derivative-free method that maintains a quadratic
// interpolation model Q(x) interpolating f at m points (default m = 2n+1).
// Each step solves the trust-region subproblem min Q(x_k + d) subject to
// ||d|| <= delta and box constraints, then updates the model and radius
// based on the accuracy ratio rho.
//
// Uses Powell's BMAT/ZMAT factored interpolation system for O(m*n) model
// updates instead of O(m*p^2) SVD. ALTMOV geometry improvement is wired
// using exact Lagrange values from BMAT/ZMAT.
//
// Requires: objective<P,S> && bound_constrained<P,S>.
// No gradient is needed -- BOBYQA uses only objective evaluations.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06.
//            K&W Section 8.4 (surrogate model framework).

#include "argmin/detail/interpolation_system.h"
#include "argmin/detail/trust_region.h"
#include "argmin/detail/bound_projection.h"
#include "argmin/options/trust_region_options.h"
#include "argmin/result/step_result.h"
#include "argmin/solver/options.h"
#include "argmin/types.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace argmin
{

// Build explicit N x N Hessian from HQ (packed upper triangle) and PQ
// (implicit second derivative via outer products of interpolation points).
//
// H = HQ_matrix + sum_k pq[k] * xpt[k] * xpt[k]^T
//
// Cost: O(n^2 * m), acceptable for n < 20.
//
// Reference: Powell 2009, Section 2 (equation 2.2).
template <typename Scalar, int N>
Eigen::Matrix<Scalar, N, N> build_explicit_hessian(
    const detail::interpolation_system<Scalar, N>& sys)
{
    const int32_t n = sys.xbase.size();
    const int32_t m = sys.m_points;

    Eigen::Matrix<Scalar, N, N> H;
    if constexpr(N == Eigen::Dynamic)
        H.setZero(n, n);
    else
        H.setZero();

    // Unpack HQ upper triangle into symmetric matrix.
    int32_t ih = 0;
    for(int32_t j = 0; j < n; ++j)
    {
        for(int32_t i = 0; i <= j; ++i)
        {
            H(i, j) = sys.hq[ih];
            H(j, i) = sys.hq[ih];
            ++ih;
        }
    }

    // Add PQ outer products: sum_k pq[k] * xpt[k] * xpt[k]^T.
    for(int32_t k = 0; k < m; ++k)
    {
        if(sys.pq[k] == Scalar(0)) continue;
        auto xk = sys.xpt.col(k).head(n);
        H.template selfadjointView<Eigen::Upper>().rankUpdate(xk, sys.pq[k]);
    }
    // Copy upper to lower.
    H.template triangularView<Eigen::StrictlyLower>() =
        H.template triangularView<Eigen::StrictlyUpper>().transpose();

    return H;
}

template <int N = dynamic_dimension>
struct bobyqa_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = bobyqa_policy<M>;

    struct options_type
    {
        std::optional<double> initial_trust_radius{};             // default: auto, 10% of max bound range (Powell 2009)
        std::optional<double> final_trust_radius{};               // default: 1e-8, stopping criterion on delta (Powell 2009)
        trust_region_options trust{};                              // Embedded trust region params
        std::uint16_t stall_window{200};
        double feasibility_gate{std::numeric_limits<double>::infinity()};
    };

    options_type options{};

    template <typename P = void>
    struct state_type
    {
        const P* problem{nullptr};
        Eigen::Vector<double, N> x;
        Eigen::Vector<double, N> lower;
        Eigen::Vector<double, N> upper;
        Eigen::Vector<double, N> scale;          // scale[i] = normalized bound range (Powell 2009, PRELIM)
        Eigen::Vector<double, N> x_scaled;       // x in scaled space: x_scaled[i] = x[i] / scale[i]
        Eigen::Vector<double, N> lower_scaled;   // lower[i] / scale[i]
        Eigen::Vector<double, N> upper_scaled;   // upper[i] / scale[i]
        double objective_value{};
        detail::interpolation_system<double, N> sys;  // BMAT/ZMAT factored interpolation system
        double delta{};
        double delta_max{};
        double final_trust_radius{1e-8};
        double rho{};       // Current minimum trust radius (contracts toward rho_end)
        double rho_end{};   // Final rho target (= final_trust_radius)
        std::uint32_t iteration{0};
        std::uint16_t rescue_counter{0};  // Consecutive rho contractions without improvement (Powell 2009, RESCUE)
        bool last_improved{false};       // Whether the previous step improved the objective
        int m{};
        bool initialized{false};

        // Powell bobyqb driver state (the ntrits state machine).
        //
        // ntrits is the number of consecutive trust-region iterations since the
        // last "alternative" (geometry) iteration:
        //   ntrits > 0  -- trust-region iteration (objective is evaluated)
        //   ntrits == 0 -- alternative / ALTMOV geometry iteration (evaluated)
        //   ntrits == -1 -- short step (dnorm < 0.5*rho): NOT evaluated, routed
        //                   to a geometry refresh or a rho reduction instead.
        //
        // nfsav is the evaluation count captured at the start of work with the
        // current rho (init, after a step longer than rho, after a rescue or a
        // rho reduction). The nevals <= nfsav+2 guard sends the FIRST short step
        // after a fresh rho refresh to geometry rather than straight to a rho
        // reduction -- the bookkeeping whose absence stalls the driver.
        //
        // nevals mirrors Powell's evaluation counter (seeded to the 2n+1
        // bootstrap count so the reported total matches the reference); nresc is
        // its value at the last rescue.
        //
        // diffa/diffb/diffc hold |Q-error| at the last three interpolation
        // points; xoptsq = ||xopt||^2, reset to zero by the origin shift.
        //
        // Reference: Powell 2009, Section 5; NLopt bobyqb_ lines 2100-3053.
        int ntrits{0};
        int nfsav{0};
        int nresc{0};
        int nevals{0};

        // Counts consecutive trust iterations whose explicit-Hessian model
        // gradient dominates the least-Frobenius-norm interpolant's gradient by
        // the reference's factor of ten (Powell 2009, Section 4; NLopt bobyqb_
        // lines 2900-2923). On the third such iteration the model is replaced by
        // the least-Frobenius-norm interpolant to shed accumulated Hessian error.
        int itest{0};
        std::uint32_t nevals_reported{0};
        double diffa{0.0};
        double diffb{0.0};
        double diffc{0.0};
        double xoptsq{0.0};

        // Diagnostic: number of short-step trials whose objective evaluation was
        // deferred (dnorm < 0.5*rho, ntrits = -1). Each such trial is NOT passed
        // to the objective; the driver routes to a geometry or rho transition
        // instead. Exposed so the no-evaluation property can be pinned.
        int short_step_count{0};
    };

    // Clamp the interpolation radius so it never exceeds half of the smallest
    // finite bound range: rhobeg <= 0.5 * min_i (upper_i - lower_i). This is
    // the precondition that lets the 2n bootstrap perturbations stay inside
    // the box on every finite-bounded coordinate.
    static double clamp_radius_to_bounds(double h,
                                         const Eigen::Vector<double, N>& lower_scaled,
                                         const Eigen::Vector<double, N>& upper_scaled)
    {
        const int n = lower_scaled.size();
        for(int i = 0; i < n; ++i)
        {
            double range_i = upper_scaled[i] - lower_scaled[i];
            if(std::isfinite(range_i) && range_i < 2.0 * h)
                h = 0.5 * range_i;
        }
        return (h > 0.0) ? h : 1.0;
    }

    // Move x0 at least h inside every finite bound. Unbounded coordinates are
    // left untouched. After this, x0 +/- h stays within [lower, upper] on all
    // finite-bounded coordinates.
    static void shift_inside_bounds(Eigen::Vector<double, N>& x_scaled, double h,
                                    const Eigen::Vector<double, N>& lower_scaled,
                                    const Eigen::Vector<double, N>& upper_scaled)
    {
        const int n = x_scaled.size();
        for(int i = 0; i < n; ++i)
        {
            if(std::isfinite(lower_scaled[i]))
                x_scaled[i] = std::max(x_scaled[i], lower_scaled[i] + h);
            if(std::isfinite(upper_scaled[i]))
                x_scaled[i] = std::min(x_scaled[i], upper_scaled[i] - h);
        }
    }

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
                    const solver_options<Convergence>& opts)
    {
        const int n = problem.dimension();
        state_type<Problem> s;
        s.problem = &problem;

        s.lower = problem.lower_bounds();
        s.upper = problem.upper_bounds();

        // Variable rescaling (Powell 2009, PRELIM concept).
        // Compute scale factors from bound ranges so that the trust region
        // treats all variables equally regardless of their physical scales.
        //
        // Mixed finite/infinite bounds must be scaled coherently. Using a raw
        // 1.0 for unbounded coordinates and then normalizing by the largest
        // range collapses every unbounded coordinate to 1 / max_finite_range:
        // when some bounded variable has a range > 1 (the common case on the
        // Hock-Schittkowski set), the unbounded coordinates are then stepped
        // hundreds of times too small. Instead, use the largest finite range as
        // the reference scale for unbounded coordinates, so they share scale 1
        // with the widest bounded variable after normalization rather than
        // being suppressed.
        s.scale.resize(n);
        double ref_range = 0.0;
        for(int i = 0; i < n; ++i)
        {
            double range = s.upper[i] - s.lower[i];
            if(std::isfinite(range) && range > ref_range)
                ref_range = range;
        }
        if(ref_range <= 0.0)
            ref_range = 1.0;
        for(int i = 0; i < n; ++i)
        {
            double range = s.upper[i] - s.lower[i];
            s.scale[i] = std::isfinite(range) ? range : ref_range;
        }
        double max_scale = s.scale.maxCoeff();
        if(max_scale > 0.0)
            s.scale /= max_scale;
        // Guard against zero-range dimensions (lower == upper)
        for(int i = 0; i < n; ++i)
        {
            if(s.scale[i] < 1e-15)
                s.scale[i] = 1.0;
        }

        // Transform bounds to scaled space
        s.lower_scaled = (s.lower.array() / s.scale.array()).matrix();
        s.upper_scaled = (s.upper.array() / s.scale.array()).matrix();

        // Project x0 to feasible region, then transform to scaled space
        s.x = detail::project(x0, s.lower, s.upper);
        s.x_scaled = (s.x.array() / s.scale.array()).matrix();

        // Number of interpolation points. The BMAT/ZMAT bootstrap assembles
        // exactly the 2n+1 coordinate-perturbation system (Powell 2009), so
        // this is fixed rather than configurable: a value above 2n+1 would let
        // step() index past the assembled points, and a smaller value would
        // mis-shape the factored algebra.
        s.m = 2 * n + 1;

        // Final trust radius
        s.final_trust_radius = options.final_trust_radius.value_or(1e-8);

        // Initial trust-region radius in SCALED space (Powell 2009).
        double h = options.initial_trust_radius.value_or(0.0);
        if(h <= 0.0)
        {
            double max_range = 0.0;
            for(int i = 0; i < n; ++i)
            {
                double range_i = s.upper_scaled[i] - s.lower_scaled[i];
                if(std::isfinite(range_i))
                    max_range = std::max(max_range, range_i);
            }
            h = (max_range > 0.0) ? 0.1 * max_range : 1.0;
        }

        // Bound-safety repair (core BOBYQA guarantee: the objective is NEVER
        // evaluated outside [xl, xu]). Powell requires rhobeg <= 0.5 * min
        // finite bound range and x0 at least rhobeg inside every finite bound
        // so the 2n coordinate perturbations of the bootstrap stay feasible.
        // Repair rather than reject: clamp h on tight ranges and shift x0
        // inward, which keeps near-bound or tight-box starts solvable instead
        // of tripping a domain error on log/sqrt objectives.
        h = clamp_radius_to_bounds(h, s.lower_scaled, s.upper_scaled);
        shift_inside_bounds(s.x_scaled, h, s.lower_scaled, s.upper_scaled);
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();

        s.delta = h;
        s.delta_max = 10.0 * h;

        // Bootstrap the BMAT/ZMAT interpolation system.
        // The bootstrap evaluates f at 2n+1 coordinate-perturbation points
        // around x_scaled and initializes BMAT, ZMAT, GOPT, HQ, PQ.
        //
        // Reference: Powell 2009, Section 2.
        //   Adapted from NLopt prelim_() lines 1710-1950.
        //   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1710
        s.sys = detail::bootstrap_interpolation_system<double, N>(
            s.x_scaled, h, s.lower_scaled, s.upper_scaled,
            [&](const Eigen::Vector<double, N>& x_sc) {
                return s.problem->value(
                    (x_sc.array() * s.scale.array()).matrix());
            });

        // Update x to the best point found during bootstrap.
        s.x_scaled = s.sys.xbase + s.sys.xopt;
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();
        s.objective_value = s.sys.fval[s.sys.kopt];

        s.initialized = true;
        s.iteration = 0;

        // Powell 2009, Section 5: two-radius rho contraction scheme.
        s.rho = s.delta;
        s.rho_end = s.final_trust_radius;

        // Seed the ntrits driver state. The bootstrap has evaluated the 2n+1
        // interpolation points, so nevals starts there (matching the reference
        // counter, which includes PRELIM). nfsav/nresc start at the same count.
        s.nevals = s.m;
        s.nevals_reported = 0;
        s.nfsav = s.m;
        s.nresc = s.m;
        s.ntrits = 0;
        s.diffa = 0.0;
        s.diffb = 0.0;
        s.diffc = 0.0;
        s.xoptsq = s.sys.xopt.squaredNorm();

        return s;
    }

    // Diagonal element of the ZMAT-factored H block: hdiag_k = sum_jj zmat(k,jj)^2.
    // This is Powell's alpha for point k (bobyqb_ lines 2503-2510).
    template <typename Sys>
    static double zmat_hdiag(const Sys& sys, int k, int nptm)
    {
        double h = 0.0;
        for(int jj = 0; jj < nptm; ++jj)
            h += sys.zmat(k, jj) * sys.zmat(k, jj);
        return h;
    }

    // Outcome of Powell's relative update-point selection.
    struct replacement_choice
    {
        int knew{0};        // index of the point to delete
        double denom{0.0};  // the chosen update denominator
        bool refuse{true};  // true when the denominator is roundoff-collapsed
    };

    // Powell's relative scaden/biglsq update-point selection for a trust-region
    // step (bobyqb_ lines 2493-2549). For each candidate k != kopt,
    //   den = beta*hdiag_k + vlag[k]^2,
    //   temp = max(1, (distsq_k / delta^2)^2)   (fourth power of distance),
    // knew maximizes the weighted denominator temp*den, and biglsq tracks the
    // largest weighted Lagrange value temp*vlag[k]^2. The update is REFUSED when
    // scaden <= 0.5*biglsq -- the chosen denominator is dominated by a Lagrange
    // value, signaling a roundoff collapse; the caller then rescues instead of
    // applying a corrupt update. The best point kopt is never a candidate, and a
    // failed selection is flagged by refuse rather than silently returning an
    // arbitrary index.
    //
    // ref_point is the point distances are measured from: xopt for the
    // pre-evaluation selection, or xnew for the post-evaluation recompute.
    template <typename Sys, typename VlagVec>
    static replacement_choice select_replacement_relative(
        const Sys& sys, const VlagVec& vlag, double beta, double delta,
        const Eigen::Vector<double, N>& ref_point)
    {
        const int nn = sys.xbase.size();
        const int mm = sys.m_points;
        const int nptm = mm - nn - 1;
        const double delsq = delta * delta;

        replacement_choice c;
        double scaden = 0.0;
        double biglsq = 0.0;
        for(int k = 0; k < mm; ++k)
        {
            if(k == sys.kopt) continue;
            double hdiag = zmat_hdiag(sys, k, nptm);
            double den = beta * hdiag + vlag[k] * vlag[k];
            double distsq = (sys.xpt.col(k).head(nn) - ref_point).squaredNorm();
            double ratio_sq = distsq / delsq;
            double temp = std::max(1.0, ratio_sq * ratio_sq);
            if(temp * den > scaden)
            {
                scaden = temp * den;
                c.knew = k;
                c.denom = den;
            }
            biglsq = std::max(biglsq, temp * vlag[k] * vlag[k]);
        }
        c.refuse = (scaden <= 0.5 * biglsq);
        return c;
    }

    // Outcome of the least-Frobenius-norm model-reset test.
    struct frobenius_reset_result
    {
        double gqsq{0.0};        // projected magnitude of the current model gradient
        double gisq{0.0};        // projected magnitude of the interpolant gradient
        bool reset_applied{false};  // whether gopt/pq/hq were replaced this call
    };

    // Powell's least-Frobenius-norm model reset (bobyqb_ lines 2824-2923).
    //
    // Forms the least-Frobenius-norm interpolant to the current data and takes
    // its gradient at xopt. It then compares the projected magnitude of the
    // current explicit-Hessian model gradient (gqsq, using sys.gopt) against the
    // interpolant's (gisq), projecting out the components a bound-active xopt
    // would push into its bound. The itest counter increments while the current
    // model dominates by the reference's factor of ten and resets otherwise;
    // after three consecutive dominations the current model's Hessian has
    // accumulated enough error that the interpolant is the better model, so
    // gopt/pq/hq are replaced by it and itest is reset.
    //
    // Reads bmat/zmat/xpt/fval/gopt/xopt; writes gopt/pq/hq only on reset. sl/su
    // are the bound offsets lower_scaled - xbase and upper_scaled - xbase.
    template <typename Sys>
    static frobenius_reset_result frobenius_model_reset(
        Sys& sys, const Eigen::Vector<double, N>& sl,
        const Eigen::Vector<double, N>& su, int& itest)
    {
        const int n = sys.xbase.size();
        const int mm = sys.m_points;
        const int nptm = mm - n - 1;

        using vec_m =
            Eigen::Vector<double, Sys::MaxM>;

        vec_m lfvlag(mm);   // fval[k] - fval[kopt]
        vec_m w(mm);        // least-Frobenius-norm pq weights
        for(int k = 0; k < mm; ++k)
        {
            lfvlag[k] = sys.fval[k] - sys.fval[sys.kopt];
            w[k] = 0.0;
        }
        for(int j = 0; j < nptm; ++j)
        {
            double sum = 0.0;
            for(int k = 0; k < mm; ++k)
                sum += sys.zmat(k, j) * lfvlag[k];
            for(int k = 0; k < mm; ++k)
                w[k] += sum * sys.zmat(k, j);
        }

        vec_m wpq(mm);      // saved interpolant pq
        vec_m wscaled(mm);  // (xpt_k . xopt) * pq_k for the gradient sum
        for(int k = 0; k < mm; ++k)
        {
            double sum = sys.xpt.col(k).head(n).dot(sys.xopt);
            wpq[k] = w[k];
            wscaled[k] = sum * w[k];
        }

        Eigen::Vector<double, N> gi(n);  // interpolant gradient at xopt
        frobenius_reset_result r;
        for(int i = 0; i < n; ++i)
        {
            double sum = 0.0;
            for(int k = 0; k < mm; ++k)
                sum += sys.bmat(k, i) * lfvlag[k] + sys.xpt(i, k) * wscaled[k];
            if(sys.xopt[i] == sl[i])
            {
                double gq = std::min(0.0, sys.gopt[i]);
                double gc = std::min(0.0, sum);
                r.gqsq += gq * gq;
                r.gisq += gc * gc;
            }
            else if(sys.xopt[i] == su[i])
            {
                double gq = std::max(0.0, sys.gopt[i]);
                double gc = std::max(0.0, sum);
                r.gqsq += gq * gq;
                r.gisq += gc * gc;
            }
            else
            {
                r.gqsq += sys.gopt[i] * sys.gopt[i];
                r.gisq += sum * sum;
            }
            gi[i] = sum;
        }

        ++itest;
        if(r.gqsq < 10.0 * r.gisq)
            itest = 0;
        if(itest >= 3)
        {
            for(int i = 0; i < n; ++i)
                sys.gopt[i] = gi[i];
            for(int k = 0; k < mm; ++k)
                sys.pq[k] = wpq[k];
            const int nh = n * (n + 1) / 2;
            for(int i = 0; i < nh; ++i)
                sys.hq[i] = 0.0;
            itest = 0;
            r.reset_applied = true;
        }
        return r;
    }

    // One advance of Powell's bobyqb driver: run the ntrits state machine from
    // the top of a trust-region iteration (label L60) until it has evaluated
    // the objective at least once and is ready to begin the next trust-region
    // iteration, or until the calculations with the final rho are complete.
    //
    // ntrits distinguishes the three iteration kinds (see state_type): a
    // trust-region step (ntrits>0, evaluated), an ALTMOV geometry step
    // (ntrits==0, evaluated), and a short step (ntrits==-1, NOT evaluated,
    // routed to a geometry refresh or a rho reduction). The no-eval short-step
    // and rho-reduction transitions are executed inline, so a single call may
    // contract rho several times before an evaluation occurs; the reported
    // evaluation count is the true number of objective calls.
    //
    // After each trust iteration the least-Frobenius-norm model reset test runs
    // (frobenius_model_reset): a degraded explicit-Hessian model is replaced by
    // the interpolant once it dominates for three consecutive iterations.
    //
    // Reference: Powell, M. J. D. (2009), Section 5.
    //   Ported statement-for-statement from NLopt bobyqb_ lines 2100-3053.
    //   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2100
    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        const int n = s.x.size();
        const int nptm = s.m - n - 1;
        const double old_incumbent_f = s.objective_value;

        // Machine labels (the bobyqb_ goto targets, reified as a state variable).
        enum class label { l60, l90, l210, l230, l360, l650, l680 };

        // Evaluate the objective at a trial point given in shifted coordinates
        // (relative to xbase), clamping into the box so the objective is never
        // probed outside [xl, xu]. Returns {f, x in scaled-absolute coords}.
        auto eval_shifted = [&](const Eigen::Vector<double, N>& xn)
            -> std::pair<double, Eigen::Vector<double, N>>
        {
            Eigen::Vector<double, N> xabs = detail::project(
                (s.sys.xbase + xn).eval(), s.lower_scaled, s.upper_scaled);
            Eigen::Vector<double, N> xorig = (xabs.array() * s.scale.array()).matrix();
            double f = s.problem->value(xorig);
            ++s.nevals;
            return {f, xabs};
        };

        // Powell's RESCUE: rebuild the interpolation system from scratch about
        // the current best point. Used when a chosen denominator has collapsed
        // (bobyqb_ L190). Kept as the bootstrap rebuild (Powell never writes
        // behind the factorization); the rebuild's 2n+1 evaluations are counted.
        auto do_rescue = [&]()
        {
            double h = s.rho;
            Eigen::Vector<double, N> xb = s.sys.xbase + s.sys.xopt;
            h = clamp_radius_to_bounds(h, s.lower_scaled, s.upper_scaled);
            shift_inside_bounds(xb, h, s.lower_scaled, s.upper_scaled);
            s.sys = detail::bootstrap_interpolation_system<double, N>(
                xb, h, s.lower_scaled, s.upper_scaled,
                [&](const Eigen::Vector<double, N>& x_sc) {
                    return s.problem->value((x_sc.array() * s.scale.array()).matrix());
                });
            s.nevals += s.m;
            s.nresc = s.nevals;
            s.nfsav = s.nevals;
            s.ntrits = 0;
            s.xoptsq = s.sys.xopt.squaredNorm();
        };

        // Within-iteration locals (never persist across calls -- each call
        // resumes at L60, the top of a fresh trust-region iteration).
        Eigen::Vector<double, N> d, xnew, gnew;
        Eigen::Vector<double, N> zero_d = Eigen::Vector<double, N>::Zero(n);
        Eigen::Vector<double, detail::interpolation_system<double, N>::MaxMpN> vlag;
        double dsq = 0.0, dnorm = 0.0, adelt = 0.0, ratio = 0.0;
        double beta = 0.0, denom = 0.0, vquad = 0.0, alpha = 0.0;
        double geo_distsq = 0.0;
        double last_eval_dnorm = 0.0;
        int knew = -1;

        bool did_eval = false;
        bool terminated = false;

        // The terminal deferred short-step evaluation (ntrits == -1) is the one
        // point Powell reports outside the factored model, and only when it
        // strictly improves on the model's best node (bobyqb_ lines 2582-2586).
        bool deferred_improved = false;
        double deferred_f = 0.0;
        Eigen::Vector<double, N> deferred_xabs;

        label loc = label::l60;
        int safety = 0;
        const int safety_cap = 200000;

        while(true)
        {
            // A terminate request (roundoff-collapsed denominator with no rescue
            // left, or the end-game short-step evaluation) is a real exit: leave
            // the driver at once rather than re-entering L60 and re-running
            // identical trust passes until the safety cap. The reference returns
            // straight out on these paths (bobyqb_ lines 2483-2484, 2546-2547,
            // 2585-2586). Without this the machine spins with no state change and
            // then reports a false convergence.
            if(terminated)
                break;
            // Return once an evaluation has been made and the driver is back at
            // the top of a trust-region iteration; keep looping through the
            // no-eval short-step / rho-reduction transitions otherwise.
            if(loc == label::l60 && did_eval)
                break;
            if(++safety > safety_cap)
            {
                terminated = true;
                break;
            }

            switch(loc)
            {
            case label::l60:
            {
                // Generate the trust-region step (TRSBOX) that minimizes the
                // model subject to the trust radius and the box.
                Eigen::Vector<double, N> mg = detail::model_gradient_at(s.sys, zero_d);
                Eigen::Matrix<double, N, N> H = build_explicit_hessian<double, N>(s.sys);
                Eigen::Vector<double, N> x_opt_abs = s.sys.xbase + s.sys.xopt;
                d = detail::solve_trust_region_box(
                    mg, H, x_opt_abs, s.delta, s.lower_scaled, s.upper_scaled);
                dsq = d.squaredNorm();
                dnorm = std::min(s.delta, std::sqrt(dsq));

                if(dnorm < 0.5 * s.rho)
                {
                    // Short step: defer the evaluation (bobyqb_ L60, lines
                    // 2157-2206). Route to a geometry refresh (L650) or a rho
                    // reduction (L680) instead of evaluating this trial.
                    s.ntrits = -1;
                    ++s.short_step_count;
                    geo_distsq = (10.0 * s.rho) * (10.0 * s.rho);

                    bool go_650 = false;
                    if(s.nevals <= s.nfsav + 2)
                    {
                        // First short step after a fresh rho refresh: refresh
                        // the geometry rather than declaring the rho complete.
                        go_650 = true;
                    }
                    else
                    {
                        double errbig = std::max({s.diffa, s.diffb, s.diffc});
                        double frhosq = 0.125 * s.rho * s.rho;
                        // crvmin is avoided: use the policy-visible minimum
                        // diagonal model curvature as the errbig-test proxy
                        // (the same per-coordinate curvature the bdtol loop
                        // below uses). Keeps the port confined to the policy.
                        double crvmin = std::numeric_limits<double>::infinity();
                        for(int j = 0; j < n; ++j)
                            crvmin = std::min(crvmin, H(j, j));
                        if(crvmin > 0.0 && errbig > frhosq * crvmin)
                        {
                            go_650 = true;
                        }
                        else
                        {
                            // Per-coordinate boundary/curvature test (bobyqb_
                            // lines 2179-2204): would freeing a bound-active
                            // coordinate, plus its curvature over rho, still
                            // predict improvement?
                            double bdtol = errbig / s.rho;
                            gnew = detail::model_gradient_at(s.sys, d);
                            Eigen::Vector<double, N> sl = s.lower_scaled - s.sys.xbase;
                            Eigen::Vector<double, N> su = s.upper_scaled - s.sys.xbase;
                            xnew = s.sys.xopt + d;
                            // The reference's TRSBOX returns a bound-active
                            // component bitwise equal to sl/su via
                            // median(sl, xopt+d, su) (bobyqb_/trsbox_), so its
                            // xnew[j] == sl[j] tests fire exactly. This port
                            // reconstructs xnew = xopt + d from the box solver's
                            // displacement, where (xopt[j] + d[j]) only reproduces
                            // (lower_scaled[j] - xbase[j]) bitwise when xopt[j] is
                            // zero; a bound-active coordinate otherwise misses the
                            // equality by a few ULPs and the curvature backstop is
                            // silently skipped. Snap to the bound within an
                            // eps-scale tolerance to recover the reference's
                            // bound-active detection (this local xnew feeds only
                            // the test; the evaluated step is rebuilt below).
                            for(int j = 0; j < n; ++j)
                            {
                                double bscale = std::max({1.0, std::abs(sl[j]),
                                                          std::abs(su[j]),
                                                          std::abs(xnew[j])});
                                // An eps-scale relative tolerance (8 ULP against
                                // the coordinate magnitude); empirically the
                                // detection is insensitive to this factor across
                                // several orders of magnitude.
                                double snap_tol =
                                    8.0 * std::numeric_limits<double>::epsilon()
                                    * bscale;
                                if(std::abs(xnew[j] - sl[j]) <= snap_tol)
                                    xnew[j] = sl[j];
                                else if(std::abs(xnew[j] - su[j]) <= snap_tol)
                                    xnew[j] = su[j];
                            }
                            for(int j = 0; j < n; ++j)
                            {
                                double bdtest = bdtol;
                                if(xnew[j] == sl[j]) bdtest = gnew[j];
                                if(xnew[j] == su[j]) bdtest = -gnew[j];
                                if(bdtest < bdtol)
                                {
                                    bdtest += 0.5 * H(j, j) * s.rho;
                                    if(bdtest < bdtol)
                                    {
                                        go_650 = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // Retain xnew for the deferred evaluation at end-game.
                    xnew = s.sys.xopt + d;
                    loc = go_650 ? label::l650 : label::l680;
                    break;
                }

                ++s.ntrits;
                loc = label::l90;
                break;
            }

            case label::l90:
            {
                // Origin shift (bobyqb_ L90, lines 2215-2316): when the trust
                // centre has drifted far from xbase, severe cancellation occurs
                // in the model and Lagrange functions. Shift xbase to xopt so
                // xopt becomes zero, re-centering BMAT/ZMAT/HQ/XPT. This runs
                // BEFORE the VLAG/BETA computation at L230 so the step is
                // expressed in the shifted frame; performing it afterward would
                // corrupt beta.
                //
                // sl/su are recomputed on the fly as lower_scaled - xbase, so
                // the sl -= xopt / su -= xopt bookkeeping is automatic. The
                // displacement d is invariant under the shift, so the trial
                // point in the new frame is simply xopt(=0) + d.
                if(dsq <= s.xoptsq * 1e-3)
                {
                    detail::shift_xbase(s.sys);
                    s.xoptsq = 0.0;
                }
                xnew = s.sys.xopt + d;
                loc = (s.ntrits == 0) ? label::l210 : label::l230;
                break;
            }

            case label::l210:
            {
                // ALTMOV geometry step for the KNEW-th point (chosen at L650).
                Eigen::Vector<double, N> geo = detail::altmov_geometry_step<double, N>(
                    s.sys, knew, adelt, s.lower_scaled - s.sys.xbase,
                    s.upper_scaled - s.sys.xbase);
                Eigen::Vector<double, N> geo_abs = detail::project(
                    (s.sys.xbase + geo).eval(), s.lower_scaled, s.upper_scaled);
                xnew = geo_abs - s.sys.xbase;
                d = xnew - s.sys.xopt;
                loc = label::l230;
                break;
            }

            case label::l230:
            {
                auto vb = detail::compute_vlag_beta(s.sys, d);
                vlag = vb.vlag;
                beta = vb.beta;

                if(s.ntrits == 0)
                {
                    // Geometry step: refuse when the denominator has collapsed
                    // relative to the Lagrange value at the point being moved
                    // (bobyqb_ line 2477).
                    alpha = zmat_hdiag(s.sys, knew, nptm);
                    denom = detail::compute_denom(vlag[knew], alpha, beta);
                    if(denom <= 0.5 * vlag[knew] * vlag[knew])
                    {
                        if(s.nevals > s.nresc) { do_rescue(); loc = label::l60; break; }
                        terminated = true;
                        loc = label::l60;
                        break;
                    }
                    loc = label::l360;
                    break;
                }

                // Trust step: choose the point to delete by Powell's relative
                // scaden/biglsq test, measuring distances from xopt.
                {
                    replacement_choice c = select_replacement_relative(
                        s.sys, vlag, beta, s.delta, s.sys.xopt);
                    knew = c.knew;
                    denom = c.denom;
                    if(c.refuse)
                    {
                        if(s.nevals > s.nresc) { do_rescue(); loc = label::l60; break; }
                        terminated = true;
                        loc = label::l60;
                        break;
                    }
                }
                loc = label::l360;
                break;
            }

            case label::l360:
            {
                auto [f, xabs] = eval_shifted(xnew);
                did_eval = true;
                last_eval_dnorm = dnorm;

                if(s.ntrits == -1)
                {
                    // The deferred short step is evaluated exactly once, at
                    // end-game (bobyqb_ lines 2582-2586). Powell returns this
                    // freshly evaluated point as the solution when it improves
                    // on the model's best node (fsave < fval[kopt]); otherwise
                    // the model incumbent is reported. Stash the improving point
                    // so the paid evaluation is not wasted.
                    if(f < s.sys.fval[s.sys.kopt])
                    {
                        deferred_improved = true;
                        deferred_f = f;
                        deferred_xabs = xabs;
                    }
                    terminated = true;
                    loc = label::l60;
                    break;
                }

                const double old_fopt = s.sys.fval[s.sys.kopt];
                vquad = detail::evaluate_interpolation_model(s.sys, d);
                double diff = f - old_fopt - vquad;
                s.diffc = s.diffb;
                s.diffb = s.diffa;
                s.diffa = std::abs(diff);
                if(dnorm > s.rho)
                    s.nfsav = s.nevals;

                if(s.ntrits > 0)
                {
                    if(vquad >= 0.0)
                    {
                        // The trust step failed to reduce the model. Powell
                        // returns here (bobyqb_ lines 2632-2637) because his
                        // exact TRSBOX only produces a non-descent step at a
                        // genuine roundoff limit. argmin's approximate XBDI-CG
                        // box solver can return a non-descent step at a
                        // non-stationary point, so aborting the whole solve
                        // would stop short of the optimum. Treat it instead as
                        // a failed trust iteration: shrink delta and let the
                        // geometry / rho-reduction path carry progress (rho
                        // exhaustion remains the true termination).
                        ratio = 0.0;
                        s.delta = std::min(0.5 * s.delta, dnorm);
                        if(s.delta <= 1.5 * s.rho)
                            s.delta = s.rho;
                    }
                    else
                    {
                        ratio = (f - old_fopt) / vquad;
                        if(ratio <= 0.1)
                            s.delta = std::min(0.5 * s.delta, dnorm);
                        else if(ratio <= 0.7)
                            s.delta = std::max(0.5 * s.delta, dnorm);
                        else
                            s.delta = std::max(0.5 * s.delta, 2.0 * dnorm);
                        if(s.delta <= 1.5 * s.rho)
                            s.delta = s.rho;
                    }

                    // Recompute knew/denom now that the new (improving) point is
                    // known, measuring distances from xnew instead of xopt
                    // (bobyqb_ lines 2656-2707). Keep the pre-evaluation
                    // selection (ksav/densav) if the recomputed one collapses.
                    // The recompute skips kopt (unlike the reference, which
                    // includes it) so the shared model-update helper's kopt/xopt
                    // move stays valid; the old best is kept as a node instead of
                    // being ejected -- a benign point-set difference.
                    if(f < old_fopt)
                    {
                        int ksav = knew;
                        double densav = denom;
                        replacement_choice c = select_replacement_relative(
                            s.sys, vlag, beta, s.delta, xnew);
                        if(c.refuse)
                        {
                            knew = ksav;
                            denom = densav;
                        }
                        else
                        {
                            knew = c.knew;
                            denom = c.denom;
                        }
                    }
                }

                detail::update_bmat_zmat(s.sys, vlag, beta, denom, knew);
                detail::update_model_on_replacement(s.sys, xnew, f, knew, d);

                bool improved_here = (f < old_fopt);
                if(improved_here)
                    s.xoptsq = s.sys.xopt.squaredNorm();

                if(s.ntrits > 0)
                {
                    // Least-Frobenius-norm model reset (bobyqb_ lines 2824-2923):
                    // when the explicit-Hessian model's gradient dominates the
                    // interpolant's for three consecutive trust iterations, shed
                    // the accumulated Hessian error by adopting the interpolant.
                    Eigen::Vector<double, N> sl = s.lower_scaled - s.sys.xbase;
                    Eigen::Vector<double, N> su = s.upper_scaled - s.sys.xbase;
                    frobenius_model_reset(s.sys, sl, su, s.itest);
                }

                if(s.ntrits == 0)
                {
                    loc = label::l60;
                    break;
                }
                if(f <= old_fopt + 0.1 * vquad)
                {
                    loc = label::l60;
                    break;
                }
                geo_distsq = std::max((2.0 * s.delta) * (2.0 * s.delta),
                                      (10.0 * s.rho) * (10.0 * s.rho));
                loc = label::l650;
                break;
            }

            case label::l650:
            {
                // Choose the interpolation point farthest from xopt beyond the
                // geometry threshold (bobyqb_ lines 2946-2963).
                knew = 0;
                bool found = false;
                double best = geo_distsq;
                for(int k = 0; k < s.m; ++k)
                {
                    double sum = (s.sys.xpt.col(k).head(n) - s.sys.xopt).squaredNorm();
                    if(sum > best)
                    {
                        best = sum;
                        knew = k;
                        found = true;
                    }
                }

                if(found)
                {
                    double dist = std::sqrt(best);
                    if(s.ntrits == -1)
                    {
                        s.delta = std::min(0.1 * s.delta, 0.5 * dist);
                        if(s.delta <= 1.5 * s.rho)
                            s.delta = s.rho;
                    }
                    s.ntrits = 0;
                    adelt = std::max(std::min(0.1 * dist, s.delta), s.rho);
                    dsq = adelt * adelt;
                    loc = label::l90;
                    break;
                }

                if(s.ntrits == -1)
                    loc = label::l680;
                else if(ratio > 0.0)
                    loc = label::l60;
                else if(std::max(s.delta, dnorm) > s.rho)
                    loc = label::l60;
                else
                    loc = label::l680;
                break;
            }

            case label::l680:
            {
                if(s.rho > s.rho_end)
                {
                    // The calculations with this rho are complete: contract rho
                    // and delta and start a fresh trust-region round.
                    std::tie(s.rho, s.delta) = detail::contract_rho(s.rho, s.rho_end);
                    s.ntrits = 0;
                    s.nfsav = s.nevals;
                    loc = label::l60;
                    break;
                }
                if(s.ntrits == -1)
                {
                    // Evaluate the deferred short Newton step once at end-game.
                    loc = label::l360;
                    break;
                }
                terminated = true;
                loc = label::l60;
                break;
            }
            }
        }

        // Keep the reported incumbent tied to the model's best node, except on
        // the terminal deferred short step that strictly improved: there Powell
        // returns the freshly evaluated point (the one solution reported outside
        // the model), so the paid evaluation carries the reported result.
        if(deferred_improved)
        {
            s.x_scaled = deferred_xabs;
            s.x = (s.x_scaled.array() * s.scale.array()).matrix();
            s.objective_value = deferred_f;
        }
        else
        {
            s.x_scaled = s.sys.xbase + s.sys.xopt;
            s.x = (s.x_scaled.array() * s.scale.array()).matrix();
            s.objective_value = s.sys.fval[s.sys.kopt];
        }

        ++s.iteration;
        bool improved = s.objective_value < old_incumbent_f;
        s.last_improved = improved;

        Eigen::Vector<double, N> gopt_now = detail::model_gradient_at(s.sys, zero_d);
        double grad_proxy = gopt_now.norm();
        if(s.rho > s.rho_end)
            grad_proxy = std::max(grad_proxy, 1.0);

        double obj_change = s.objective_value - old_incumbent_f;
        double effective_step = improved ? last_eval_dnorm : s.delta;
        double effective_change = improved ? obj_change : s.delta;

        std::uint32_t evals = static_cast<std::uint32_t>(s.nevals) - s.nevals_reported;
        s.nevals_reported = static_cast<std::uint32_t>(s.nevals);

        step_result<double> r{
            .objective_value = s.objective_value,
            .gradient_norm = grad_proxy,
            .step_size = effective_step,
            .objective_change = effective_change,
            .improved = improved,
            .x_norm = s.x.norm(),
        };
        // Report the true number of objective calls this step made. A terminate
        // path that exits without evaluating (a refused, unrescuable denominator)
        // legitimately makes zero calls; fabricating a phantom evaluation there
        // would corrupt the accumulated evaluation total.
        r.evaluations = evals;
        if(terminated)
            r.policy_status = solver_status::converged;
        return r;
    }

    // Cold restart -- BOBYQA has no warm-start mode since the interpolation
    // set is point-specific.
    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        reset_clear(s, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        s.x = detail::project(x0, s.lower, s.upper);
        s.x_scaled = (s.x.array() / s.scale.array()).matrix();
        s.iteration = 0;
        s.initialized = false;

        double h = s.delta;
        h = clamp_radius_to_bounds(h, s.lower_scaled, s.upper_scaled);
        shift_inside_bounds(s.x_scaled, h, s.lower_scaled, s.upper_scaled);
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();

        s.sys = detail::bootstrap_interpolation_system<double, N>(
            s.x_scaled, h, s.lower_scaled, s.upper_scaled,
            [&](const Eigen::Vector<double, N>& x_sc) {
                return s.problem->value(
                    (x_sc.array() * s.scale.array()).matrix());
            });

        s.x_scaled = s.sys.xbase + s.sys.xopt;
        s.x = (s.x_scaled.array() * s.scale.array()).matrix();
        s.objective_value = s.sys.fval[s.sys.kopt];
        s.initialized = true;
    }
};

}

#endif
