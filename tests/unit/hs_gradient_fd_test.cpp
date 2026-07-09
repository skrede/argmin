// Hock-Schittkowski analytic-derivative FD verification.
//
// The HS analytic gradients (and constraint Jacobians) were never checked
// against an independent numerical reference -- a subtly wrong analytic
// derivative could silently mask or reward a solver defect. This file pins
// every HS problem's analytic derivatives against the Ridders numerical
// harness (tests/unit/fd_verification.h) and includes a dedicated
// seeded-wrong-gradient case proving the check is sensitive to a corrupted
// derivative.
//
// References:
//   Hock, W. & Schittkowski, K. (1981) "Test Examples for Nonlinear
//     Programming Codes", Lecture Notes in Economics and Mathematical
//     Systems 187, Springer.
//   Ridders, C. J. F. (1982) "Accurate computation of F'(x) and F''(x)",
//     Advances in Engineering Software 4(2), 75-76.

#include "fd_verification.h"

#include "argmin/test_functions/hock_schittkowski.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <array>
#include <cstddef>

using namespace argmin;

namespace
{

// Deterministic, asymmetric perturbation pattern -- avoids coincidental
// cancellation that a symmetric offset could hide. Values are small enough
// to stay near each problem's initial point (no HS objective in this
// registry has a variable-dependent sqrt/log domain restriction, so no
// feasibility check beyond "near the given start" is required).
constexpr std::array<double, 5> perturbation_pattern{0.11, -0.07, 0.13, -0.09, 0.05};

template <typename Scalar, int N>
[[nodiscard]] auto perturbation(int n) -> Eigen::Vector<Scalar, N>
{
    Eigen::Vector<Scalar, N> d(n);
    for(int i = 0; i < n; ++i)
        d[i] = static_cast<Scalar>(perturbation_pattern[static_cast<std::size_t>(i)
                                                          % perturbation_pattern.size()]);
    return d;
}

// FD-verify one problem's analytic gradient (and, where present, its
// constraint Jacobian) at several interior points against the Ridders
// numerical reference. ~1e-8 gate: tight enough to catch a wrong term or
// sign, loose enough to absorb Ridders' own ~1e-10 extrapolation error.
template <typename Problem>
void check_hs_gradient_and_jacobian()
{
    constexpr int N = Problem::problem_dimension;
    constexpr int M = Problem::constraint_count;
    constexpr double gate = 1e-8;

    Problem p;
    const Eigen::Vector<double, N> x0 = p.initial_point();
    const Eigen::Vector<double, N> d = perturbation<double, N>(x0.size());

    const std::array<Eigen::Vector<double, N>, 3> points{
        x0, Eigen::Vector<double, N>(x0 + d), Eigen::Vector<double, N>(x0 - d)};

    for(const auto& x : points)
    {
        Eigen::Vector<double, N> g_analytic(x.size());
        p.gradient(x, g_analytic);
        const double disc = argmin_test::fd::gradient_discrepancy(p, x, g_analytic);
        INFO("objective gradient max discrepancy: " << disc);
        CHECK(disc < gate);

        if constexpr(M > 0)
        {
            Eigen::MatrixXd J_analytic(M, N);
            p.constraint_jacobian(x, J_analytic);

            Eigen::MatrixXd J_numeric;
            argmin_test::fd::ridders_jacobian(
                [&](const Eigen::Vector<double, N>& xx, Eigen::VectorXd& cc)
                {
                    p.constraints(xx, cc);
                },
                x, M, J_numeric);

            double max_disc = 0.0;
            for(int r = 0; r < M; ++r)
                for(int c = 0; c < N; ++c)
                {
                    const double denom = std::max(1.0, std::abs(J_numeric(r, c)));
                    max_disc = std::max(max_disc,
                                         std::abs(J_analytic(r, c) - J_numeric(r, c)) / denom);
                }
            INFO("constraint jacobian max discrepancy: " << max_disc);
            CHECK(max_disc < gate);
        }
    }
}

}

TEMPLATE_TEST_CASE("hs_gradient_fd: analytic derivatives match Ridders numerical reference",
                    "[hs_gradient_fd]",
                    hs001<>, hs002<>, hs005<>, hs006<>, hs007<>, hs021<>, hs023<>, hs024<>,
                    hs026<>, hs027<>, hs028<>, hs029<>, hs030<>, hs031<>, hs034<>, hs035<>,
                    hs036<>, hs037<>, hs038<>, hs039<>, hs040<>, hs043<>, hs044<>, hs048<>,
                    hs050<>, hs051<>, hs052<>, hs071<>, hs076<>)
{
    check_hs_gradient_and_jacobian<TestType>();
}

namespace
{

// A COPY of hs001's gradient with a deliberate single-term error (the
// "-2*(1-x0)" term is dropped from g[0]). hock_schittkowski.h is never
// touched; this wrapper exists solely to prove the FD check is sensitive
// to a wrong analytic gradient.
struct hs001_corrupted_gradient
{
    hs001<> inner;
    static constexpr int problem_dimension = hs001<>::problem_dimension;

    [[nodiscard]] double value(const Eigen::Vector<double, problem_dimension>& x) const
    {
        return inner.value(x);
    }

    void gradient(const Eigen::Vector<double, problem_dimension>& x,
                  Eigen::Vector<double, problem_dimension>& g) const
    {
        const double t = x[1] - x[0] * x[0];
        g[0] = -400.0 * x[0] * t;   // missing "- 2*(1 - x[0])" term
        g[1] = 200.0 * t;
    }
};

}

TEST_CASE("hs_gradient_fd: seeded wrong gradient is caught by the FD check", "[hs_gradient_fd][!shouldfail]")
{
    // Expected to FAIL: the corrupted gradient must not pass the same gate
    // that every genuine HS gradient passes above. [!shouldfail] reports
    // this as the expected disposition (a red-state proof), not a build
    // break.
    hs001_corrupted_gradient corrupted;
    const auto x0 = corrupted.inner.initial_point();
    Eigen::Vector2d g_wrong;
    corrupted.gradient(x0, g_wrong);
    const double disc = argmin_test::fd::gradient_discrepancy(corrupted, x0, g_wrong);
    INFO("corrupted-gradient discrepancy: " << disc);
    CHECK(disc < 1e-8);
}

TEST_CASE("hs_gradient_fd: correct hs001 gradient passes the same FD check", "[hs_gradient_fd]")
{
    hs001<> correct;
    const auto x0 = correct.initial_point();
    Eigen::Vector2d g_correct;
    correct.gradient(x0, g_correct);
    const double disc = argmin_test::fd::gradient_discrepancy(correct, x0, g_correct);
    INFO("correct-gradient discrepancy: " << disc);
    CHECK(disc < 1e-8);
}
