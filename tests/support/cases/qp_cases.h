#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_QP_CASES_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_CASES_QP_CASES_H

#include "cases/case_options.h"

#include "argmin/expected.h"

#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"
#include "argmin/qp/sparse_admm_qp.h"
#include "argmin/qp/dense_active_set_qp.h"

#include "argmin/options/dense_qp_options.h"
#include "argmin/options/sparse_qp_options.h"
#include "argmin/options/dense_active_set_qp_options.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <array>
#include <limits>
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace argmin::cases
{

inline constexpr std::array<std::string_view, 3> qp_keys{"dense_admm_qp", "sparse_admm_qp", "dense_active_set_qp"};

// min 0.5 x^T P x + q^T x subject to l <= A x <= u. The start is feasible, as the active-set method
// requires; the operator-splitting solvers ignore it.
struct qp_case
{
    Eigen::MatrixXd P;
    Eigen::VectorXd q;
    Eigen::MatrixXd A;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
    Eigen::VectorXd start;
};

inline qp_case qp_with_rows(Eigen::Index rows)
{
    qp_case c{Eigen::MatrixXd(3, 3), Eigen::VectorXd(3), Eigen::MatrixXd(rows, 3), Eigen::VectorXd(rows), Eigen::VectorXd(rows), Eigen::VectorXd(3)};
    c.P << 4.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0;
    c.q << 1.0, 1.0, -1.0;
    c.start << 0.5, 0.5, 0.0;
    return c;
}

// Both lower bounds on x0 and x1 and the upper bound on x2 are active at the solution (0, 0, 0.25).
inline qp_case bound_qp()
{
    qp_case c = qp_with_rows(3);
    c.A.setIdentity();
    c.l << 0.0, 0.0, 0.0;
    c.u << 1.0, 1.0, 0.25;
    c.start << 0.0, 0.0, 0.0;
    return c;
}

inline qp_case inequality_qp()
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    qp_case c = qp_with_rows(3);
    c.A << 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0;
    c.l << 1.0, -inf, -inf;
    c.u << inf, 0.5, 0.75;
    return c;
}

inline qp_case equality_qp()
{
    qp_case c = qp_with_rows(2);
    c.A << 1.0, 1.0, 0.0, 0.0, 1.0, 1.0;
    c.l << 1.0, 0.5;
    c.u = c.l;
    return c;
}

// The equality row x0 + x1 = 1 and the upper bound on x2 are active at the solution.
inline qp_case mixed_qp()
{
    qp_case c = qp_with_rows(4);
    c.A << 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    c.l << 1.0, 0.0, 0.0, 0.0;
    c.u << 1.0, 0.7, 0.7, 0.25;
    return c;
}

inline expected<qp_result<double>, qp_error> solve_qp(dense_admm_qp_solver<double>& solver, const qp_case& c, const dense_qp_options<double>& options)
{
    return solver.solve(c.P, c.q, c.A, c.l, c.u, options);
}

inline expected<qp_result<double>, qp_error> solve_qp(sparse_admm_qp_solver<double>& solver, const qp_case& c, const sparse_qp_options<double>& options)
{
    const Eigen::SparseMatrix<double> P = c.P.sparseView();
    const Eigen::SparseMatrix<double> A = c.A.sparseView();
    return solver.solve(P, c.q, A, c.l, c.u, options);
}

inline expected<qp_result<double>, qp_error> solve_qp(dense_active_set_qp_solver<double>& solver, const qp_case& c, const dense_active_set_qp_options<double>& options)
{
    solver.warm_start(c.start, Eigen::VectorXd());
    return solver.solve(c.P, c.q, c.A, c.l, c.u, options);
}

template <typename Solver>
Solver make_qp_solver(const qp_case& c)
{
    if constexpr(std::is_constructible_v<Solver, int, int>)
        return Solver(static_cast<int>(c.P.rows()), static_cast<int>(c.A.rows()));
    else
        return Solver();
}

template <typename Solver, typename Visitor, typename Options>
void visit_qp_classes(Visitor& visitor, std::string_view key, const Options& options)
{
    const std::array<qp_case, 4> data{bound_qp(), inequality_qp(), equality_qp(), mixed_qp()};
    const std::array<case_id, 4> ids{{{key, "bound", "bound_qp"}, {key, "inequality", "inequality_qp"}, {key, "equality", "equality_qp"}, {key, "mixed", "mixed_qp"}}};
    for(std::size_t i = 0; i < data.size(); ++i)
        visitor(ids[i], data[i], options, [&]() { return make_qp_solver<Solver>(data[i]); });
}

template <typename Visitor>
void visit_qp_cases(Visitor& visitor)
{
    visit_qp_classes<dense_admm_qp_solver<double>>(visitor, "dense_admm_qp", dense_qp_options<double>{});
    visit_qp_classes<sparse_admm_qp_solver<double>>(visitor, "sparse_admm_qp", sparse_qp_options<double>{});
    visit_qp_classes<dense_active_set_qp_solver<double>>(visitor, "dense_active_set_qp", dense_active_set_qp_options<double>{});
}

}

#endif
