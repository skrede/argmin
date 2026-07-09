#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_BENCHMARK_RESULT_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_BENCHMARK_RESULT_H

#include "argmin/test_functions/problem_class.h"

#include <format>
#include <string>
#include <cstdint>
#include <string_view>

namespace argmin::bench
{

struct benchmark_result
{
    std::string_view solver;
    std::string_view library;
    std::string_view problem;
    problem_class pclass{};
    int dimension{};
    std::uint64_t seed{};
    std::string_view mode;
    int solver_iters{};
    int f_evals{};
    int g_evals{};
    int c_evals{};
    int J_evals{};
    std::int64_t wall_time_us{};
    double final_objective{};
    double known_optimum{};
    double accuracy{};
    double constraint_violation{};
    std::string_view status;
    std::string row_disposition{"included"};
    std::string cap_status{"none"};
    std::string exclusion_reason;
    std::int64_t solve_wall_time_us{};
    std::int64_t end_to_end_wall_time_us{};
    std::string provenance_id;

    // Returned decision vector and, when the solver state carries them,
    // the constraint multipliers at the final iterate. Both are
    // semicolon-delimited packed float fields serialized into the
    // returned-point sidecar (publish_returned_points.csv), NOT into the
    // fixed-width publication summary schema -- the vector length varies by
    // problem, so it cannot occupy fixed summary columns. multipliers is
    // empty when the policy state does not expose them; the independent
    // validator then recovers a least-squares multiplier estimate from the
    // analytic constraint gradients.
    std::string returned_point;
    std::string multipliers;
};

[[nodiscard]] inline auto csv_header() -> std::string_view
{
    return "solver,library,problem,class,dimension,seed,mode,solver_iters,"
           "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
           "known_optimum,accuracy,constraint_violation,status,row_disposition,"
           "cap_status,exclusion_reason,solve_wall_time_us,end_to_end_wall_time_us,"
           "provenance_id";
}

[[nodiscard]] inline auto csv_row(const benchmark_result& r) -> std::string
{
    return std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{:.15e},{:.15e},{:.15e},{:.15e},{},{},{},{},{},{},{}",
                       r.solver, r.library, r.problem,
                       to_string(r.pclass), r.dimension,
                       r.seed, r.mode, r.solver_iters,
                       r.f_evals, r.g_evals, r.c_evals, r.J_evals,
                       r.wall_time_us,
                       r.final_objective, r.known_optimum,
                       r.accuracy, r.constraint_violation, r.status,
                       r.row_disposition, r.cap_status, r.exclusion_reason,
                       r.solve_wall_time_us, r.end_to_end_wall_time_us,
                       r.provenance_id);
}

}

#endif
