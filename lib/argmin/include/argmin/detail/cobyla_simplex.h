#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_SIMPLEX_H

// Simplex construction and geometry maintenance for COBYLA.
//
// COBYLA (Constrained Optimization BY Linear Approximation) maintains a
// simplex of n+1 vertices in R^n. At each iteration, linear models of
// the objective and constraints are built by interpolating at the simplex
// vertices. These functions handle simplex construction, linear model
// fitting, vertex replacement, and geometry maintenance.
//
// Reference: Powell, M. J. D. (1994) "A direct search optimization
//            method that models the objective and constraint functions
//            by linear interpolation."
//            K&W 2e, Section 10.7 (derivative-free constrained optimization).

#include "argmin/detail/bound_projection.h"
#include "argmin/types.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace argmin::detail
{

// Result of building linear models from simplex interpolation.
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
struct cobyla_linear_models
{
    Eigen::Vector<Scalar, N> objective_gradient;
    Eigen::Matrix<Scalar, M, N> constraint_gradients;
};

// Build initial simplex from x0 by perturbing each coordinate by rho.
//
// Constructs n+1 vertices: x0 plus n perturbations along coordinate
// directions, respecting bounds. Uses the same bound-aware perturbation
// as BOBYQA's initial interpolation set.
//
// Returns matrix of shape (n, n+1) where each column is a vertex.
//
// Reference: Powell 1994, Section 2 (initial simplex construction).
template <typename Scalar = double, int N = argmin::dynamic_dimension>
Eigen::Matrix<Scalar, N, Eigen::Dynamic> build_simplex(
    const Eigen::Vector<Scalar, N>& x0,
    Scalar rho,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    const int n = static_cast<int>(x0.size());
    Eigen::Matrix<Scalar, N, Eigen::Dynamic> simplex(n, n + 1);
    simplex.col(0) = x0;

    for(int i = 0; i < n; ++i)
    {
        Eigen::Vector<Scalar, N> pt = x0;
        const Scalar upward_room   = upper[i] - x0[i];
        const Scalar downward_room = x0[i] - lower[i];

        // Pre-fix this used `pt[i] = min(pt[i] + rho, upper[i])` and
        // only flipped to -rho when the upward step was clipped to
        // *zero* (`abs(...) < 1e-15 * rho`). For an x0 close to but
        // not at the upper bound (e.g. x0[i] = upper[i] - 0.3 * rho)
        // the simplex got an asymmetric 0.3 * rho displacement,
        // 3x ill-scaled with no trigger. Static-audit C8.
        //
        // Choose +rho if there is full upward room; else -rho if there
        // is full downward room; else fall back to whichever bound
        // gives the larger displacement (move all the way to that
        // bound). This guarantees a displacement of at least
        // min(rho, max(upward_room, downward_room)).
        if(upward_room >= rho)
            pt[i] = x0[i] + rho;
        else if(downward_room >= rho)
            pt[i] = x0[i] - rho;
        else if(upward_room >= downward_room)
            pt[i] = upper[i];
        else
            pt[i] = lower[i];

        simplex.col(1 + i) = pt;
    }

    return simplex;
}

// Stateful simplex model builder with pre-allocated workspace.
//
// Pre-allocates the displacement matrix, function deltas, and LU workspace
// used by build_linear_models to eliminate per-step dynamic allocation.
//
// Reference: Powell 1994, Section 3 (linear interpolation).

template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
class cobyla_simplex_solver
{
public:
    explicit cobyla_simplex_solver(int n, int m_total)
        : n_{n}
        , m_{m_total}
        , D_(n, n)
        , df_(n)
        , dc_(m_total, n)
    {
        models_.objective_gradient.resize(n);
        models_.constraint_gradients.resize(m_total, n);
        vertex_map_.reserve(static_cast<std::size_t>(n));
    }

    cobyla_simplex_solver() = default;

    // Build linear models of objective and constraints by interpolation.
    //
    // Reference: Powell 1994, Section 3 (linear interpolation).
    template <int P = argmin::dynamic_dimension>
    const cobyla_linear_models<Scalar, N, M>& build_models(
        const Eigen::Matrix<Scalar, N, P>& simplex,
        const Eigen::Vector<Scalar, P>& f_values,
        const Eigen::Matrix<Scalar, M, P>& c_values,
        int best_idx)
    {
        Eigen::Vector<Scalar, N> x_best = simplex.col(best_idx);
        Scalar f_best = f_values[best_idx];

        int col = 0;
        for(int i = 0; i <= n_; ++i)
        {
            if(i == best_idx)
                continue;
            D_.col(col) = simplex.col(i) - x_best;
            df_[col] = f_values[i] - f_best;
            if(m_ > 0)
                dc_.col(col) = c_values.col(i) - c_values.col(best_idx);
            ++col;
        }

        auto lu = D_.transpose().fullPivLu();

        models_.objective_gradient = lu.solve(df_);

        if(m_ > 0)
        {
            for(int j = 0; j < m_; ++j)
                models_.constraint_gradients.row(j) = lu.solve(
                    Eigen::Vector<Scalar, N>(dc_.row(j).transpose())).transpose();
        }

        return models_;
    }

    // Check simplex geometry by examining the displacement matrix condition.
    //
    // Returns the index of the vertex contributing most to degeneracy if the
    // condition number exceeds the threshold, or -1 if geometry is acceptable.
    //
    // Reference: Powell 1994, Section 5 (geometry maintenance).
    int check_geometry(
        const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& simplex,
        int best_idx,
        Scalar rho)
    {
        Eigen::Vector<Scalar, N> x_best = simplex.col(best_idx);

        vertex_map_.clear();
        int col = 0;
        for(int i = 0; i <= n_; ++i)
        {
            if(i == best_idx)
                continue;
            D_.col(col) = simplex.col(i) - x_best;
            vertex_map_.push_back(i);
            ++col;
        }

        auto lu = D_.fullPivLu();
        Scalar cond = Scalar(1) / lu.rcond();

        // Threshold tightened from 1e10 to 1e7: the prior value tolerated
        // near-singular D and let the linear-model gradients drift on
        // poorly-scaled simplices. 1e7 is conservative for double-
        // precision; below that, linear-model interpolation gradients
        // remain accurate to roughly 1e-9 relative. Static-audit C11.
        if(cond > Scalar(1e7) || !std::isfinite(cond))
        {
            int worst = vertex_map_[0];
            Scalar min_dist = (simplex.col(worst) - x_best).squaredNorm();

            for(int i = 1; i < n_; ++i)
            {
                int vi = vertex_map_[static_cast<std::size_t>(i)];
                Scalar dist = (simplex.col(vi) - x_best).squaredNorm();
                if(dist < min_dist)
                {
                    min_dist = dist;
                    worst = vi;
                }
            }
            return worst;
        }

        (void)rho;
        return -1;
    }

private:
    int n_{0};
    int m_{0};
    Eigen::Matrix<Scalar, N, N> D_;
    Eigen::Vector<Scalar, N> df_;
    Eigen::Matrix<Scalar, M, N> dc_;
    cobyla_linear_models<Scalar, N, M> models_;
    std::vector<int> vertex_map_;
};

// Backward-compatible free function: build linear models.
//
// Reference: Powell 1994, Section 3 (linear interpolation).
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension,
          int P = argmin::dynamic_dimension>
cobyla_linear_models<Scalar, N, M> build_linear_models(
    const Eigen::Matrix<Scalar, N, P>& simplex,
    const Eigen::Vector<Scalar, P>& f_values,
    const Eigen::Matrix<Scalar, M, P>& c_values,
    int best_idx)
{
    const int n = static_cast<int>(simplex.rows());
    const int m = static_cast<int>(c_values.rows());
    cobyla_simplex_solver<Scalar, N, M> solver(n, m);
    return solver.build_models(simplex, f_values, c_values, best_idx);
}

// Replace vertex k in the simplex with a new point.
//
// Reference: Powell 1994, Section 4 (vertex replacement).
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension,
          int P = argmin::dynamic_dimension>
void replace_vertex(
    Eigen::Matrix<Scalar, N, P>& simplex,
    Eigen::Vector<Scalar, P>& f_values,
    Eigen::Matrix<Scalar, M, P>& c_values,
    int k,
    const Eigen::Vector<Scalar, N>& x_new,
    Scalar f_new,
    const Eigen::Vector<Scalar, M>& c_new)
{
    simplex.col(k) = x_new;
    f_values[k] = f_new;
    if(c_values.rows() > 0)
        c_values.col(k) = c_new;
}

// Select which vertex to replace when inserting a new trial point.
//
// Replaces the vertex farthest from x_new, excluding the best vertex.
// This approximates Powell's criterion of maximising the absolute
// determinant of the interpolation matrix.
//
// Reference: Powell 1994, Section 4.
template <typename Scalar = double, int N = argmin::dynamic_dimension, int P = argmin::dynamic_dimension>
int select_replacement_vertex(
    const Eigen::Matrix<Scalar, N, P>& simplex,
    const Eigen::Vector<Scalar, P>& f_values,
    int best_idx,
    const Eigen::Vector<Scalar, N>& x_new)
{
    const int m = static_cast<int>(simplex.cols());

    int farthest = (best_idx == 0) ? 1 : 0;
    Scalar max_dist = (simplex.col(farthest) - x_new).squaredNorm();

    for(int i = 0; i < m; ++i)
    {
        if(i == best_idx)
            continue;
        Scalar dist = (simplex.col(i) - x_new).squaredNorm();
        if(dist > max_dist)
        {
            max_dist = dist;
            farthest = i;
        }
    }

    return farthest;
}

// Backward-compatible free function: check simplex geometry.
//
// Reference: Powell 1994, Section 5 (geometry maintenance).
template <typename Scalar = double, int N = argmin::dynamic_dimension, int P = argmin::dynamic_dimension>
int check_simplex_geometry(
    const Eigen::Matrix<Scalar, N, P>& simplex,
    int best_idx,
    Scalar rho)
{
    const int n = static_cast<int>(simplex.rows());
    cobyla_simplex_solver<Scalar, N> solver(n, 0);
    return solver.check_geometry(simplex, best_idx, rho);
}

// Generate a geometry-improving replacement for degenerate vertex k.
//
// Places the new point at x_best + rho * e_j where j is chosen to
// maximise the displacement matrix determinant, projected to bounds.
//
// Reference: Powell 1994, Section 5.
template <typename Scalar = double, int N = argmin::dynamic_dimension>
Eigen::Vector<Scalar, N> geometry_improving_point(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& simplex,
    int best_idx,
    int k,
    Scalar rho,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper)
{
    const int n = static_cast<int>(simplex.rows());
    Eigen::Vector<Scalar, N> x_best = simplex.col(best_idx);

    int col_idx = 0;
    int j = 0;
    for(int i = 0; i <= n; ++i)
    {
        if(i == best_idx)
            continue;
        if(i == k)
        {
            j = col_idx;
            break;
        }
        ++col_idx;
    }

    j = std::min(j, n - 1);

    Eigen::Vector<Scalar, N> pt = x_best;
    pt[j] = std::min(pt[j] + rho, upper[j]);
    if(std::abs(pt[j] - x_best[j]) < Scalar(1e-15) * rho)
        pt[j] = std::max(x_best[j] - rho, lower[j]);

    return project<Scalar, N>(pt, lower, upper);
}

}

#endif
