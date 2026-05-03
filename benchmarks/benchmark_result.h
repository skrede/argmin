#ifndef HPP_GUARD_ARGMIN_BENCHMARKS_BENCHMARK_RESULT_H
#define HPP_GUARD_ARGMIN_BENCHMARKS_BENCHMARK_RESULT_H

#include "argmin/test_functions/problem_class.h"

#include <cstdint>
#include <format>
#include <string>
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
    std::string_view status;
};

[[nodiscard]] inline auto csv_header() -> std::string_view
{
    return "solver,library,problem,class,dimension,seed,mode,solver_iters,"
           "f_evals,g_evals,c_evals,J_evals,wall_time_us,final_objective,"
           "known_optimum,accuracy,status";
}

[[nodiscard]] inline auto csv_row(const benchmark_result& r) -> std::string
{
    return std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{:.15e},{:.15e},{:.15e},{}",
                       r.solver, r.library, r.problem,
                       to_string(r.pclass), r.dimension,
                       r.seed, r.mode, r.solver_iters,
                       r.f_evals, r.g_evals, r.c_evals, r.J_evals,
                       r.wall_time_us,
                       r.final_objective, r.known_optimum,
                       r.accuracy, r.status);
}

}

#endif
