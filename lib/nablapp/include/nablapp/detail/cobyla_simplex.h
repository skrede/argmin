#ifndef HPP_GUARD_NABLAPP_DETAIL_COBYLA_SIMPLEX_H
#define HPP_GUARD_NABLAPP_DETAIL_COBYLA_SIMPLEX_H

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

#include "nablapp/detail/bound_projection.h"
#include "nablapp/types.h"

#include <Eigen/Core>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nablapp::detail
{

// Result of building linear models from simplex interpolation.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
struct cobyla_linear_models
{
    Eigen::Vector<Scalar, N> objective_gradient;
    Eigen::MatrixX<Scalar> constraint_gradients;  // m_total x n
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
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
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
        pt[i] = std::min(pt[i] + rho, upper[i]);

        if(std::abs(pt[i] - x0[i]) < Scalar(1e-15) * rho)
            pt[i] = std::max(x0[i] - rho, lower[i]);

        simplex.col(1 + i) = pt;
    }

    return simplex;
}

// Build linear models of objective and constraints by interpolation.
//
// Given n+1 simplex points with objective values f and constraint values
// c (m_total x (n+1)), fits linear models:
//   L_0(x) = f(x_best) + g_obj^T (x - x_best)
//   L_j(x) = c_j(x_best) + g_cj^T (x - x_best)
//
// The displacement matrix D = [v_1 - x_best, ..., v_n - x_best] is n x n
// (using all non-best vertices). Solves D^T g = delta_f for each model.
//
// Reference: Powell 1994, Section 3 (linear interpolation).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
cobyla_linear_models<Scalar, N> build_linear_models(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& simplex,
    const Eigen::VectorX<Scalar>& f_values,
    const Eigen::MatrixX<Scalar>& c_values,
    int best_idx)
{
    const int n = static_cast<int>(simplex.rows());
    const int m = static_cast<int>(c_values.rows());

    Eigen::Vector<Scalar, N> x_best = simplex.col(best_idx);
    Scalar f_best = f_values[best_idx];

    // Build displacement matrix (n x n) and function deltas
    Eigen::Matrix<Scalar, N, N> D(n, n);
    Eigen::Vector<Scalar, N> df(n);
    Eigen::MatrixX<Scalar> dc(m, n);

    int col = 0;
    for(int i = 0; i <= n; ++i)
    {
        if(i == best_idx)
            continue;
        D.col(col) = simplex.col(i) - x_best;
        df[col] = f_values[i] - f_best;
        if(m > 0)
            dc.col(col) = c_values.col(i) - c_values.col(best_idx);
        ++col;
    }

    // Solve D^T g = delta via LU decomposition
    auto lu = D.transpose().fullPivLu();

    cobyla_linear_models<Scalar, N> models;
    models.objective_gradient = lu.solve(df);

    if(m > 0)
    {
        models.constraint_gradients.resize(m, n);
        for(int j = 0; j < m; ++j)
            models.constraint_gradients.row(j) = lu.solve(dc.row(j).transpose()).transpose();
    }
    else
    {
        models.constraint_gradients.resize(0, n);
    }

    return models;
}

// Replace vertex k in the simplex with a new point.
//
// Reference: Powell 1994, Section 4 (vertex replacement).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
void replace_vertex(
    Eigen::Matrix<Scalar, N, Eigen::Dynamic>& simplex,
    Eigen::VectorX<Scalar>& f_values,
    Eigen::MatrixX<Scalar>& c_values,
    int k,
    const Eigen::Vector<Scalar, N>& x_new,
    Scalar f_new,
    const Eigen::VectorX<Scalar>& c_new)
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
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
int select_replacement_vertex(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& simplex,
    const Eigen::VectorX<Scalar>& f_values,
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

// Check simplex geometry by examining the displacement matrix condition.
//
// Returns the index of the vertex contributing most to degeneracy if the
// condition number exceeds the threshold, or -1 if geometry is acceptable.
//
// Reference: Powell 1994, Section 5 (geometry maintenance).
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
int check_simplex_geometry(
    const Eigen::Matrix<Scalar, N, Eigen::Dynamic>& simplex,
    int best_idx,
    Scalar rho)
{
    const int n = static_cast<int>(simplex.rows());
    Eigen::Vector<Scalar, N> x_best = simplex.col(best_idx);

    // Build displacement matrix
    Eigen::Matrix<Scalar, N, N> D(n, n);
    std::vector<int> vertex_map(static_cast<std::size_t>(n));
    int col = 0;
    for(int i = 0; i <= n; ++i)
    {
        if(i == best_idx)
            continue;
        D.col(col) = simplex.col(i) - x_best;
        vertex_map[static_cast<std::size_t>(col)] = i;
        ++col;
    }

    auto lu = D.fullPivLu();
    Scalar cond = Scalar(1) / lu.rcond();

    if(cond > Scalar(1e10) || !std::isfinite(cond))
    {
        // Find the vertex closest to x_best (most degenerate)
        int worst = vertex_map[0];
        Scalar min_dist = (simplex.col(worst) - x_best).squaredNorm();

        for(int i = 1; i < n; ++i)
        {
            int vi = vertex_map[static_cast<std::size_t>(i)];
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

// Generate a geometry-improving replacement for degenerate vertex k.
//
// Places the new point at x_best + rho * e_j where j is chosen to
// maximise the displacement matrix determinant, projected to bounds.
//
// Reference: Powell 1994, Section 5.
template <typename Scalar = double, int N = nablapp::dynamic_dimension>
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

    // Determine which coordinate direction vertex k corresponded to.
    // Use the direction that maximises the distance from x_best.
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

    // Clamp j to valid range
    j = std::min(j, n - 1);

    Eigen::Vector<Scalar, N> pt = x_best;
    pt[j] = std::min(pt[j] + rho, upper[j]);
    if(std::abs(pt[j] - x_best[j]) < Scalar(1e-15) * rho)
        pt[j] = std::max(x_best[j] - rho, lower[j]);

    return project<Scalar, N>(pt, lower, upper);
}

}

#endif
