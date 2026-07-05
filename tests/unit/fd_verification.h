#ifndef HPP_GUARD_ARGMIN_TESTS_UNIT_FD_VERIFICATION_H
#define HPP_GUARD_ARGMIN_TESTS_UNIT_FD_VERIFICATION_H

// Test-only numerical-derivative verification harness (Ridders extrapolation
// primary, complex-step optional). Used to FD-verify analytic gradients
// against a reference that cannot share a bug with the analytic code path.
//
// Not shipped: lives under tests/, never included from argmin/argmin.h or
// any production header.
//
// Ridders extrapolation builds a Neville tableau of central-difference
// estimates at geometrically shrinking step sizes and extrapolates toward
// h -> 0, tracking an error estimate at each stage and stopping once it
// stops improving. Real evaluations only -- no complex/dual re-templating
// of the objective is required, so it works on any Problem already
// instantiated on a concrete Scalar (e.g. double).
// Reference: Ridders, C. J. F. (1982) "Accurate computation of F'(x) and
//   F''(x)", Advances in Engineering Software 4(2), 75-76; Press, Teukolsky,
//   Vetterling & Flannery, Numerical Recipes 3rd ed., Sec. 5.7
//   ("Derivatives").
//
// Complex-step Im f(x+ih)/h (Squire & Trapp 1998) is exposed as an OPTIONAL
// path for callables genuinely templated on Scalar (no subtractive
// cancellation, ~1e-12 accurate). Objectives fixed to a concrete Scalar
// (e.g. the Hock-Schittkowski problems as instantiated on double) cannot
// use it without re-templating, which this harness deliberately avoids --
// Ridders above is the primary gate for those.

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <vector>
#include <complex>
#include <concepts>
#include <type_traits>

namespace argmin_test::fd
{

// Ridders extrapolation for the derivative of a scalar callable f(t) at
// t = 0 (callers wrap f(t) = objective(x + t*e_i) to get one gradient
// component). n_tab rows of Neville extrapolation; step shrink factor con
// and safety factor safe follow the standard Numerical Recipes dfridr
// parameters and stopping rule (bail out once the error estimate worsens
// by more than safe between successive tableau diagonals).
template <typename F>
[[nodiscard]] auto ridders_derivative(F&& f, double h0 = 1e-2, int n_tab = 10,
                                       double con = 1.4, double safe = 2.0) -> double
{
    const double con2 = con * con;
    std::vector<std::vector<double>> a(n_tab, std::vector<double>(n_tab, 0.0));

    double hh = h0;
    a[0][0] = (f(hh) - f(-hh)) / (2.0 * hh);
    double err = std::numeric_limits<double>::max();
    double ans = a[0][0];

    for(int i = 1; i < n_tab; ++i)
    {
        hh /= con;
        a[0][i] = (f(hh) - f(-hh)) / (2.0 * hh);
        double fac = con2;
        for(int j = 1; j <= i; ++j)
        {
            a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - 1.0);
            fac *= con2;
            const double errt = std::max(std::abs(a[j][i] - a[j - 1][i]),
                                          std::abs(a[j][i] - a[j - 1][i - 1]));
            if(errt <= err)
            {
                err = errt;
                ans = a[j][i];
            }
        }
        if(std::abs(a[i][i] - a[i - 1][i - 1]) >= safe * err) break;
    }
    return ans;
}

// Numerical gradient of p.value(x) via per-coordinate Ridders
// extrapolation. ~1e-10 accurate on the smooth double-precision HS
// objectives; real evaluations only, no re-templating of Problem.
template <typename Problem, typename Scalar, int N>
void ridders_gradient(const Problem& p, const Eigen::Vector<Scalar, N>& x,
                       Eigen::Vector<Scalar, N>& g)
{
    const int n = static_cast<int>(x.size());
    g.resize(n);
    Eigen::Vector<Scalar, N> xp = x;
    for(int i = 0; i < n; ++i)
    {
        const Scalar xi = x[i];
        g[i] = static_cast<Scalar>(ridders_derivative([&](double t)
        {
            xp[i] = xi + static_cast<Scalar>(t);
            return static_cast<double>(p.value(xp));
        }));
        xp[i] = xi;
    }
}

// Numerical Jacobian of a vector-valued callable f(x, out) -- the
// constraints(x, c) / residuals(x, r) convention -- via per-entry Ridders
// extrapolation. num_outputs rows, x.size() columns.
template <typename VectorFunction, typename Scalar, int N>
void ridders_jacobian(VectorFunction&& f, const Eigen::Vector<Scalar, N>& x,
                       int num_outputs, Eigen::MatrixX<Scalar>& J)
{
    const int n = static_cast<int>(x.size());
    J.resize(num_outputs, n);
    Eigen::Vector<Scalar, N> xp = x;
    Eigen::VectorX<Scalar> c(num_outputs);
    for(int j = 0; j < n; ++j)
    {
        const Scalar xj = x[j];
        for(int k = 0; k < num_outputs; ++k)
        {
            J(k, j) = static_cast<Scalar>(ridders_derivative([&](double t)
            {
                xp[j] = xj + static_cast<Scalar>(t);
                f(xp, c);
                return static_cast<double>(c[k]);
            }));
        }
        xp[j] = xj;
    }
}

// Max componentwise absolute/relative discrepancy between an analytic and
// a numerical gradient: relative where the numerical component is not
// tiny, absolute otherwise -- avoids a false red state at a near-zero
// stationary component.
template <typename Scalar, int N>
[[nodiscard]] auto max_gradient_discrepancy(const Eigen::Vector<Scalar, N>& analytic,
                                             const Eigen::Vector<Scalar, N>& numerical) -> Scalar
{
    Scalar worst = Scalar(0);
    for(int i = 0; i < analytic.size(); ++i)
    {
        const Scalar abs_err = std::abs(analytic[i] - numerical[i]);
        const Scalar denom = std::max(Scalar(1), std::abs(numerical[i]));
        worst = std::max(worst, abs_err / denom);
    }
    return worst;
}

// Single entry point: given a problem, a point, and its analytic gradient,
// returns the max discrepancy against the Ridders numerical gradient.
template <typename Problem, typename Scalar, int N>
[[nodiscard]] auto gradient_discrepancy(const Problem& p, const Eigen::Vector<Scalar, N>& x,
                                         const Eigen::Vector<Scalar, N>& analytic_gradient) -> Scalar
{
    Eigen::Vector<Scalar, N> numerical(x.size());
    ridders_gradient(p, x, numerical);
    return max_gradient_discrepancy(analytic_gradient, numerical);
}

// OPTIONAL: complex-step derivative Im f(x+ih)/h, ~1e-12, no subtractive
// cancellation. Only usable when f is genuinely templated on Scalar (i.e.
// invocable with std::complex<double>); Ridders above is the primary gate
// for objectives already fixed to a concrete Scalar.
// Reference: Squire, W. & Trapp, G. (1998) "Using Complex Variables to
//   Estimate Derivatives of Real Functions", SIAM Review 40(1), 110-112.
template <typename F>
concept complex_step_capable = requires(F&& f, std::complex<double> z)
{
    { f(z) } -> std::convertible_to<std::complex<double>>;
};

template <typename F>
    requires complex_step_capable<F>
[[nodiscard]] auto complex_step_derivative(F&& f, double h = 1e-20) -> double
{
    return std::imag(f(std::complex<double>(0.0, h))) / h;
}

}

#endif
