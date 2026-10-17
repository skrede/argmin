#include "argmin/qp/qp_types.h"
#include "argmin/qp/sparse_admm_qp.h"

#include "mm_data/mm_problems.h"
#include "sparse_control_qp_family.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace argmin;

namespace
{

using sparse = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using row_major = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

const char* status_name(qp_solve_status s)
{
    switch(s)
    {
    case qp_solve_status::solved: return "solved";
    case qp_solve_status::solved_inaccurate: return "solved_inaccurate";
    case qp_solve_status::max_iterations: return "max_iterations";
    case qp_solve_status::primal_infeasible: return "primal_infeasible";
    case qp_solve_status::dual_infeasible: return "dual_infeasible";
    }
    return "unknown";
}

struct record
{
    std::string group;
    std::string label;
    int n{0};
    int m{0};
    int nnz_a{0};
    double density{0.0};
    bool solved_call{false};
    qp_solve_status status{qp_solve_status::solved};
    int iterations{0};
    bool polished{false};
    double primal_residual{0.0};
    double dual_residual{0.0};
    double objective{0.0};
    bool has_optimum{false};
    double objective_gap{0.0};
};

double stored_density(const sparse& A)
{
    const double cells = static_cast<double>(A.rows()) * static_cast<double>(A.cols());
    return cells > 0.0 ? static_cast<double>(A.nonZeros()) / cells : 0.0;
}

record measure(const char* group, const std::string& label, const sparse& P,
               const Eigen::VectorXd& q, const sparse& A, const Eigen::VectorXd& l,
               const Eigen::VectorXd& u, bool has_opt, double optimum)
{
    sparse_admm_qp_solver<double> solver;
    auto result = solver.solve(P, q, A, l, u);

    record r;
    r.group = group;
    r.label = label;
    r.n = static_cast<int>(A.cols());
    r.m = static_cast<int>(A.rows());
    r.nnz_a = static_cast<int>(A.nonZeros());
    r.density = stored_density(A);
    r.has_optimum = has_opt;
    r.solved_call = result.has_value();
    if(r.solved_call)
    {
        const auto& qr = *result;
        r.status = qr.status;
        r.iterations = qr.iterations;
        r.polished = qr.polished;
        r.primal_residual = qr.primal_residual;
        r.dual_residual = qr.dual_residual;
        r.objective = qr.objective_value;
        r.objective_gap = has_opt ? std::abs(qr.objective_value - optimum) : 0.0;
    }
    return r;
}

record measure_dense_source(const char* label, int n, int m, const double* pd, const double* qd,
                            const double* ad, const double* ld, const double* ud, bool has_opt,
                            double optimum)
{
    const Eigen::MatrixXd Pd = Eigen::Map<const row_major>(pd, n, n);
    const Eigen::MatrixXd Ad = Eigen::Map<const row_major>(ad, m, n);
    return measure("committed", label, Pd.sparseView(0.0, 0.0),
                   Eigen::Map<const Eigen::VectorXd>(qd, n), Ad.sparseView(0.0, 0.0),
                   Eigen::Map<const Eigen::VectorXd>(ld, m),
                   Eigen::Map<const Eigen::VectorXd>(ud, m), has_opt, optimum);
}

}

// Ship-measure-report over the sparse operator-splitting solver: the committed
// problem set loaded into sparse form, and every member of the structured
// control-shaped family. The hard assertions are honesty properties only -- the
// call returned a value, the reported status is one of the terminal outcomes,
// and every reported quantity is finite. There is deliberately NO accuracy,
// iteration or density threshold here: accuracy enforcement lives in the
// dense-sparse parity leg (sparse_admm_qp_test.cpp) and in the independent
// first-order optimality check over the control-shaped family
// (sparse_control_qp_test.cpp). This case records behavior under the project's
// ship-measure-report posture; it is not a gate.
TEST_CASE("sparse QP ship-measure-report", "[qp][sparse_ref]")
{
    std::vector<record> records;
#define MM_PROBLEM(nm)                                                                  \
    records.push_back(measure_dense_source(                                             \
        argmin::mm_data::nm::label, argmin::mm_data::nm::n, argmin::mm_data::nm::m,     \
        argmin::mm_data::nm::P.data(), argmin::mm_data::nm::q.data(),                   \
        argmin::mm_data::nm::A.data(), argmin::mm_data::nm::l.data(),                   \
        argmin::mm_data::nm::u.data(), argmin::mm_data::nm::has_optimum,                \
        argmin::mm_data::nm::optimum));
#include "mm_data/mm_problems.inc"
#undef MM_PROBLEM

    // The generated family carries no independently verified optimum, so its
    // objective-gap column stays empty rather than being filled from the
    // solver's own answer.
    for(const auto& p : argmin::control_qp_data::control_qp_family())
        records.push_back(measure("control-shaped", "H=" + std::to_string(p.horizon), p.P, p.q,
                                  p.A, p.l, p.u, false, 0.0));

    REQUIRE_FALSE(records.empty());

    for(const auto& r : records)
    {
        INFO("problem " << r.group << "/" << r.label);
        CHECK(r.solved_call);
        CHECK((r.status == qp_solve_status::solved
               || r.status == qp_solve_status::solved_inaccurate
               || r.status == qp_solve_status::max_iterations
               || r.status == qp_solve_status::primal_infeasible
               || r.status == qp_solve_status::dual_infeasible));
        CHECK(std::isfinite(r.primal_residual));
        CHECK(std::isfinite(r.dual_residual));
        CHECK(std::isfinite(r.objective));
    }

    std::printf("\n==== BEGIN SPARSE QP MEASURED REFERENCE ====\n");
    std::printf("| group | problem | n | m | nnz(A) | density | status | iters | polished |"
                " r_p | r_d | obj | obj_gap |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for(const auto& r : records)
    {
        char gap[32];
        if(r.has_optimum)
            std::snprintf(gap, sizeof(gap), "%.3e", r.objective_gap);
        else
            std::snprintf(gap, sizeof(gap), " ");
        std::printf("| %s | %s | %d | %d | %d | %.3e | %s | %d | %s | %.3e | %.3e | %.6e | %s |\n",
                    r.group.c_str(), r.label.c_str(), r.n, r.m, r.nnz_a, r.density,
                    status_name(r.status), r.iterations, r.polished ? "yes" : "no",
                    r.primal_residual, r.dual_residual, r.objective, gap);
    }
    std::printf("==== END SPARSE QP MEASURED REFERENCE ====\n\n");
}
