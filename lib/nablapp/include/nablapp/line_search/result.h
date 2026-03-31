#ifndef HPP_GUARD_NABLAPP_LINE_SEARCH_RESULT_H
#define HPP_GUARD_NABLAPP_LINE_SEARCH_RESULT_H

namespace nablapp
{

// Result of a line search computation.
//
// Returned by all line search algorithms (armijo, strong_wolfe, etc.).
// The caller checks `success` to determine whether the step satisfies
// the requested conditions.

template <typename Scalar = double>
struct line_search_result
{
    Scalar alpha{};
    Scalar value{};
    int evaluations{0};
    bool success{false};
};

}

#endif
