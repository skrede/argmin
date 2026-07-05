#include "argmin/detail/cauchy_point.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

using Catch::Approx;
using argmin::detail::cauchy_point_solver;

namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

// Minimal Hessian surrogate exposing the .multiply() interface the
// generalized-Cauchy-point solver consumes (a dense symmetric B stands
// in for the compact L-BFGS operator). A NON-diagonal B is the whole
// point of these pins: the Algorithm-CP derivative reconstruction at a
// breakpoint only deviates from the diagonal approximation when the
// off-diagonal curvature couples a bound coordinate into the free one.
struct dense_B
{
    Eigen::MatrixXd M;
    Eigen::VectorXd multiply(const Eigen::VectorXd& d) const
    {
        return M * d;
    }
};

bool contains(const std::vector<int>& v, int i)
{
    for(int e : v)
        if(e == i)
            return true;
    return false;
}

}

// ─── Kernel pin (finding 17a): segment-entry f'(t_old+) >= 0 ────────
//
// Reference: Byrd, Lu, Nocedal 1995 (L-BFGS-B) Algorithm CP -- when the
//            reconstructed derivative at the entry of a new segment is
//            non-negative, the generalized Cauchy point is at t_old, not
//            at the next breakpoint.
//
// Instance (all quantities hand-derived from the Algorithm-CP
// recurrences): x = (0, 0), g = (2, 1), lower = (-1, -3), no upper
// bounds, B = [[1, 0.7], [0.7, 1]] (non-diagonal, SPD). Breakpoints:
// coordinate 0 at t = 0.5, coordinate 1 at t = 3.
//
//   d0 = -g = (-2, -1); f'(0) = g.d0 = -5; f''(0) = d0^T B d0 = 7.8.
//   Interior minimizer of segment 0 is at t* = 5/7.8 = 0.641 > 0.5, so
//   the walk crosses the first breakpoint at t = 0.5.
//   At t = 0.5 fix coordinate 0: d = (0, -1); z = x(0.5) - x = (-1, -0.5).
//   Reconstruct f'(0.5+) = g.d + z^T (B d) = -1 + 1.2 = +0.2 >= 0.
//   => the GCP is at t_old = 0.5: x(0.5) = (-1, -0.5), with coordinate 0
//      at its lower bound (active) and coordinate 1 FREE.
//
// Pre-fix code omits the f'(t_old+) >= 0 test at segment entry, adds
// dt*f'' and returns at the NEXT breakpoint (t = 3): x_cauchy projects
// to (-1, -3) with coordinate 1 wrongly driven to its bound. Recorded
// pre-fix red: GCP = (-1, -3), coordinate 1 active.
TEST_CASE("cauchy_point segment-entry f' >= 0 stops the GCP at t_old",
          "[kernel-pin][cauchy_point][segment-entry]")
{
    Eigen::Vector2d x;
    x << 0.0, 0.0;
    Eigen::Vector2d g;
    g << 2.0, 1.0;
    Eigen::Vector2d lower;
    lower << -1.0, -3.0;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::MatrixXd M(2, 2);
    M << 1.0, 0.7,
         0.7, 1.0;
    dense_B B{M};

    cauchy_point_solver<double, 2> solver(2);
    const auto& res = solver.solve(x, g, lower, upper, B);

    INFO("x_cauchy = (" << res.x_cauchy[0] << ", " << res.x_cauchy[1]
         << ")  coord1 free = " << contains(res.free_indices, 1));
    CHECK(res.x_cauchy[0] == Approx(-1.0).margin(1e-12));
    CHECK(res.x_cauchy[1] == Approx(-0.5).margin(1e-12));
    CHECK(contains(res.free_indices, 1));
    CHECK(contains(res.active_indices, 0));
}

// ─── Kernel pin (finding 17b): final-segment minimization ──────────
//
// Reference: Byrd, Lu, Nocedal 1995 Algorithm CP -- after the last
//            breakpoint, if f' < 0 the GCP includes the final-segment
//            minimizer delta_t* = -f'/f'' along the remaining free
//            direction.
//
// Instance (hand-derived): x = (0, 0), g = (1, -1), lower0 = -0.5,
// coordinate 1 unbounded (g1 < 0, upper1 = +inf => no breakpoint),
// B = I. Single breakpoint: coordinate 0 at t = 0.5.
//
//   d0 = -g = (-1, 1); f'(0) = -2; f''(0) = 2; interior t* = 1 > 0.5.
//   At t = 0.5 fix coordinate 0: d = (0, 1); z = (-0.5, 0.5);
//   f'(0.5+) = g.d + z^T (B d) = -1 + 0.5 = -0.5 < 0, f'' = 1.
//   Final-segment minimizer delta_t* = -f'/f'' = 0.5, so the GCP is
//   x(0.5) + delta_t* * d = (-0.5, 0.5) + (0, 0.5) = (-0.5, 1.0).
//
// Pre-fix code returns x(t_last) = (-0.5, 0.5) without the final
// minimization. Recorded pre-fix red: GCP coordinate 1 = 0.5.
TEST_CASE("cauchy_point final-segment minimization after last breakpoint",
          "[kernel-pin][cauchy_point][final-segment]")
{
    Eigen::Vector2d x;
    x << 0.0, 0.0;
    Eigen::Vector2d g;
    g << 1.0, -1.0;
    Eigen::Vector2d lower;
    lower << -0.5, -kInf;
    Eigen::Vector2d upper;
    upper << kInf, kInf;

    Eigen::MatrixXd M = Eigen::MatrixXd::Identity(2, 2);
    dense_B B{M};

    cauchy_point_solver<double, 2> solver(2);
    const auto& res = solver.solve(x, g, lower, upper, B);

    INFO("x_cauchy = (" << res.x_cauchy[0] << ", " << res.x_cauchy[1]
         << ")");
    CHECK(res.x_cauchy[0] == Approx(-0.5).margin(1e-12));
    CHECK(res.x_cauchy[1] == Approx(1.0).margin(1e-12));
}
