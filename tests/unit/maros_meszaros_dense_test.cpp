#include "argmin/qp/qp_types.h"
#include "argmin/qp/dense_admm_qp.h"

#include "mm_data/mm_problems.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace argmin;

namespace
{

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
    std::string label;
    int n{0};
    int m{0};
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

using row_major = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

record measure(const char* label, int n, int m, const double* pd, const double* qd,
               const double* ad, const double* ld, const double* ud, bool has_opt,
               double optimum)
{
    const Eigen::MatrixXd P = Eigen::Map<const row_major>(pd, n, n);
    const Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(qd, n);
    const Eigen::MatrixXd A = Eigen::Map<const row_major>(ad, m, n);
    const Eigen::VectorXd l = Eigen::Map<const Eigen::VectorXd>(ld, m);
    const Eigen::VectorXd u = Eigen::Map<const Eigen::VectorXd>(ud, m);

    dense_admm_qp_solver<double> solver(n, m);
    auto result = solver.solve(P, q, A, l, u);

    record r;
    r.label = label;
    r.n = n;
    r.m = m;
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

}

// Ship-measure-report over the committed dense convex-QP subset. The solver is
// run once per problem with polish on; status, iterations, polish flag, unscaled
// residuals, and the objective gap against the verified optimum are recorded and
// emitted. The hard assertions are honesty properties only -- the call returns a
// value, the reported status is one of the terminal outcomes, and every reported
// quantity is finite. There is deliberately NO per-problem accuracy or iteration
// threshold: accuracy enforcement lives in the parity test against a trusted
// oracle; this runner records behavior under the project's empirical posture.
TEST_CASE("maros_meszaros dense-subset ship-measure-report", "[qp][mm]")
{
    std::vector<record> records;
#define MM_PROBLEM(nm)                                                             \
    records.push_back(measure(argmin::mm_data::nm::label, argmin::mm_data::nm::n,  \
                              argmin::mm_data::nm::m, argmin::mm_data::nm::P.data(),\
                              argmin::mm_data::nm::q.data(),                        \
                              argmin::mm_data::nm::A.data(),                        \
                              argmin::mm_data::nm::l.data(),                        \
                              argmin::mm_data::nm::u.data(),                        \
                              argmin::mm_data::nm::has_optimum,                     \
                              argmin::mm_data::nm::optimum));
#include "mm_data/mm_problems.inc"
#undef MM_PROBLEM

    REQUIRE_FALSE(records.empty());

    for(const auto& r : records)
    {
        INFO("problem " << r.label);
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

    std::printf("\n==== BEGIN DENSE QP MEASURED REFERENCE ====\n");
    std::printf("| problem | n | m | status | iters | polished | r_p | r_d | obj_gap |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|\n");
    for(const auto& r : records)
    {
        char gap[32];
        if(r.has_optimum)
            std::snprintf(gap, sizeof(gap), "%.3e", r.objective_gap);
        else
            std::snprintf(gap, sizeof(gap), "n/a");
        std::printf("| %s | %d | %d | %s | %d | %s | %.3e | %.3e | %s |\n",
                    r.label.c_str(), r.n, r.m, status_name(r.status), r.iterations,
                    r.polished ? "yes" : "no", r.primal_residual, r.dual_residual, gap);
    }
    std::printf("==== END DENSE QP MEASURED REFERENCE ====\n\n");
}
