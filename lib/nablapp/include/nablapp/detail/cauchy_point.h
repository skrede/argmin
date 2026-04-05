#ifndef HPP_GUARD_NABLAPP_DETAIL_CAUCHY_POINT_H
#define HPP_GUARD_NABLAPP_DETAIL_CAUCHY_POINT_H

#include "nablapp/types.h"
#include "nablapp/detail/compact_lbfgs.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace nablapp::detail
{

// Result of the generalized Cauchy point computation.

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
struct cauchy_result
{
    Eigen::Vector<Scalar, N> x_cauchy;
    std::vector<int> free_indices;    // indices NOT at bounds after GCP
    std::vector<int> active_indices;  // indices AT bounds after GCP
};

// Classify indices into free and active sets based on x_cauchy position.
// Writes into pre-allocated vectors if provided; otherwise allocates.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
void classify_indices(const Eigen::Vector<Scalar, N>& x_cauchy,
                      const Eigen::Vector<Scalar, N>& lower,
                      const Eigen::Vector<Scalar, N>& upper,
                      cauchy_result<Scalar, N>& result)
{
    const int n = x_cauchy.size();
    constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

    result.x_cauchy = x_cauchy;
    result.free_indices.clear();
    result.active_indices.clear();

    for(int i = 0; i < n; ++i)
    {
        bool at_lower = x_cauchy[i] <= lower[i] + eps * (Scalar(1) + std::abs(lower[i]));
        bool at_upper = x_cauchy[i] >= upper[i] - eps * (Scalar(1) + std::abs(upper[i]));
        if(at_lower || at_upper)
            result.active_indices.push_back(i);
        else
            result.free_indices.push_back(i);
    }
}

// Backward-compatible free function returning by value.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
cauchy_result<Scalar, N> classify_indices(const Eigen::Vector<Scalar, N>& x_cauchy,
                                          const Eigen::Vector<Scalar, N>& lower,
                                          const Eigen::Vector<Scalar, N>& upper)
{
    cauchy_result<Scalar, N> result;
    result.free_indices.reserve(x_cauchy.size());
    result.active_indices.reserve(x_cauchy.size());
    classify_indices<Scalar, N>(x_cauchy, lower, upper, result);
    return result;
}

// Stateful Cauchy point solver with pre-allocated workspace.
//
// Pre-allocates all temporaries used during the generalized Cauchy point
// computation to eliminate per-step dynamic allocation.
//
// Reference: N&W Section 16.6, pp. 475-477, eq. 16.44-16.48.
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class cauchy_point_solver
{
public:
    explicit cauchy_point_solver(int n)
        : d_(n)
        , Bd_(n)
        , x_cauchy_(n)
    {
        bps_.reserve(n);
        result_.free_indices.reserve(n);
        result_.active_indices.reserve(n);
        result_.x_cauchy.resize(n);
    }

    cauchy_point_solver() = default;

    // Compute the generalized Cauchy point using pre-allocated workspace.
    //
    // Reference: N&W Section 16.6, pp. 475-477, eq. 16.44-16.48.
    const cauchy_result<Scalar, N>& solve(
        const Eigen::Vector<Scalar, N>& x,
        const Eigen::Vector<Scalar, N>& g,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        const auto& B)
    {
        const int n = x.size();
        constexpr Scalar inf = std::numeric_limits<Scalar>::infinity();

        // Step 1: compute breakpoints (N&W eq. 16.46)
        bps_.clear();

        for(int i = 0; i < n; ++i)
        {
            Scalar ti;
            if(g[i] < Scalar(0) && upper[i] < inf)
                ti = (x[i] - upper[i]) / g[i];
            else if(g[i] > Scalar(0) && lower[i] > -inf)
                ti = (x[i] - lower[i]) / g[i];
            else
                continue;

            if(ti > Scalar(0))
                bps_.push_back({i, ti});
        }

        // Unconstrained fallback (D-04): no finite breakpoints, all variables free.
        if(bps_.empty())
        {
            d_.noalias() = -g;
            Bd_ = B.multiply(d_);
            Scalar dTBd = d_.dot(Bd_);
            Scalar gTd = g.dot(d_);

            Scalar t_star = Scalar(0);
            if(dTBd > Scalar(0))
                t_star = -gTd / dTBd;
            else if(gTd < Scalar(0))
                t_star = Scalar(1);

            result_.x_cauchy.noalias() = x + t_star * d_;
            result_.free_indices.clear();
            result_.free_indices.resize(n);
            std::iota(result_.free_indices.begin(), result_.free_indices.end(), 0);
            result_.active_indices.clear();

            return result_;
        }

        // Step 2: sort breakpoints by t
        std::sort(bps_.begin(), bps_.end(),
                  [](const auto& a, const auto& b) { return a.t < b.t; });

        // Step 3: walk the piecewise-linear path, tracking f'(t) and f''(t)
        d_.noalias() = -g;

        for(int i = 0; i < n; ++i)
        {
            if(x[i] <= lower[i] && g[i] > Scalar(0))
                d_[i] = Scalar(0);
            else if(x[i] >= upper[i] && g[i] < Scalar(0))
                d_[i] = Scalar(0);
        }

        Scalar f_prime = g.dot(d_);
        Bd_ = B.multiply(d_);
        Scalar f_double_prime = -d_.dot(Bd_);

        Scalar t_old = Scalar(0);

        for(const auto& bp : bps_)
        {
            Scalar dt = bp.t - t_old;

            if(f_double_prime > Scalar(0) && f_prime < Scalar(0))
            {
                Scalar t_star = t_old - f_prime / f_double_prime;
                if(t_star >= t_old && t_star <= bp.t)
                {
                    x_cauchy_.noalias() = x - t_star * g;
                    x_cauchy_ = project<Scalar, N>(x_cauchy_, lower, upper);
                    classify_indices<Scalar, N>(x_cauchy_, lower, upper, result_);
                    return result_;
                }
            }

            f_prime += dt * f_double_prime;

            if(f_prime >= Scalar(0))
            {
                x_cauchy_.noalias() = x - bp.t * g;
                x_cauchy_ = project<Scalar, N>(x_cauchy_, lower, upper);
                classify_indices<Scalar, N>(x_cauchy_, lower, upper, result_);
                return result_;
            }

            d_[bp.index] = Scalar(0);
            t_old = bp.t;

            Bd_ = B.multiply(d_);
            f_double_prime = -d_.dot(Bd_);

            f_prime = g.dot(d_) + t_old * (-f_double_prime);
        }

        Scalar t_last = bps_.back().t;
        x_cauchy_.noalias() = x - t_last * g;
        x_cauchy_ = project<Scalar, N>(x_cauchy_, lower, upper);
        classify_indices<Scalar, N>(x_cauchy_, lower, upper, result_);
        return result_;
    }

private:
    struct breakpoint
    {
        int index;
        Scalar t;
    };

    std::vector<breakpoint> bps_;
    Eigen::Vector<Scalar, N> d_;
    Eigen::Vector<Scalar, N> Bd_;
    Eigen::Vector<Scalar, N> x_cauchy_;
    cauchy_result<Scalar, N> result_;
};

// Backward-compatible free function wrapper.
//
// Reference: N&W Section 16.6, pp. 475-477, eq. 16.44-16.48.
//            Byrd, Lu, Nocedal, Zhu 1995 (L-BFGS-B algorithm).
template <typename Scalar = double, int N = nablapp::dynamic_dimension, int M = 7>
cauchy_result<Scalar, N> cauchy_point(
    const Eigen::Vector<Scalar, N>& x,
    const Eigen::Vector<Scalar, N>& g,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const compact_lbfgs<Scalar, N, M>& B)
{
    cauchy_point_solver<Scalar, N> solver(x.size());
    const auto& result = solver.solve(x, g, lower, upper, B);
    return result;
}

}

#endif
