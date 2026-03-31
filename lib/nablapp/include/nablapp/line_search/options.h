#ifndef HPP_GUARD_NABLAPP_LINE_SEARCH_OPTIONS_H
#define HPP_GUARD_NABLAPP_LINE_SEARCH_OPTIONS_H

namespace nablapp
{

// Options controlling line search behavior.
//
// c1: sufficient decrease parameter (Armijo condition).
//     N&W Eq. 3.6a, p. 33. Typical range [1e-4, 1e-1].
//
// c2: curvature condition parameter (Wolfe condition).
//     N&W Eq. 3.7b, p. 34. c2 = 0.9 for quasi-Newton, 0.1 for CG.
//     Must satisfy 0 < c1 < c2 < 1.
//
// rho: backtracking shrink factor for Armijo.
//
// max_alpha: initial step length (unit step for quasi-Newton methods).
//
// max_iterations: evaluation budget.

template <typename Scalar = double>
struct line_search_options
{
    Scalar c1{Scalar(1e-4)};
    Scalar c2{Scalar(0.9)};
    Scalar rho{Scalar(0.5)};
    Scalar max_alpha{Scalar(1)};
    int max_iterations{40};
};

}

#endif
