#ifndef HPP_GUARD_NABLAPP_BENCHMARKS_BENCHMARK_RESULT_H
#define HPP_GUARD_NABLAPP_BENCHMARKS_BENCHMARK_RESULT_H

#include "nablapp/test_functions/problem_class.h"

#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace nablapp::bench
{

struct benchmark_result
{
    std::string_view solver;
    std::string_view library;
    std::string_view problem;
    problem_class pclass{};
    int dimension{};
    int f_evals{};
    int g_evals{};
    std::int64_t wall_time_us{};
    double final_objective{};
    double known_optimum{};
    double accuracy{};
    std::string_view status;
};

[[nodiscard]] inline auto csv_header() -> std::string_view
{
    return "solver,library,problem,class,dimension,f_evals,g_evals,"
           "wall_time_us,final_objective,known_optimum,accuracy,status";
}

[[nodiscard]] inline auto csv_row(const benchmark_result& r) -> std::string
{
    return std::format("{},{},{},{},{},{},{},{},{:.15e},{:.15e},{:.15e},{}",
                       r.solver, r.library, r.problem,
                       to_string(r.pclass), r.dimension,
                       r.f_evals, r.g_evals, r.wall_time_us,
                       r.final_objective, r.known_optimum,
                       r.accuracy, r.status);
}

}

#endif
