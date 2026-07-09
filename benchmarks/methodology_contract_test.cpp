#include "bench_config.h"
#include "benchmark_result.h"
#include "counting_problem.h"

#include <cmath>
#include <string>
#include <iostream>
#include <string_view>

namespace
{

struct contract_problem
{
    static constexpr int problem_dimension = 1;
    static constexpr argmin::problem_class pclass = argmin::problem_class::unconstrained;

    [[nodiscard]] int dimension() const { return 1; }

    template <typename Vec>
    [[nodiscard]] double value(const Vec&) const
    {
        return 0.0;
    }

    [[nodiscard]] double optimal_value() const { return 0.0; }
};

[[nodiscard]] auto contains(std::string_view haystack,
                            std::string_view needle) -> bool
{
    return haystack.find(needle) != std::string_view::npos;
}

void require(bool condition, const char* message, int& failures)
{
    if(condition)
        return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void check_csv_contract(int& failures)
{
    const auto header = argmin::bench::csv_header();
    require(contains(header, "row_disposition"), "missing row_disposition column", failures);
    require(contains(header, "cap_status"), "missing cap_status column", failures);
    require(contains(header, "exclusion_reason"), "missing exclusion_reason column", failures);
    require(contains(header, "solve_wall_time_us"), "missing solve_wall_time_us column", failures);
    require(contains(header, "end_to_end_wall_time_us"), "missing end_to_end_wall_time_us column", failures);
    require(contains(header, "provenance_id"), "missing provenance_id column", failures);

    const argmin::bench::benchmark_result row{
        .solver = "solver",
        .library = "library",
        .problem = "problem",
        .pclass = argmin::problem_class::inequality,
        .dimension = 2,
        .seed = 7,
        .mode = "publication",
        .solver_iters = 3,
        .f_evals = 4,
        .g_evals = 5,
        .c_evals = 6,
        .J_evals = 7,
        .wall_time_us = 8,
        .final_objective = 9.0,
        .known_optimum = 10.0,
        .accuracy = 1.0,
        .constraint_violation = 1e-9,
        .status = "converged",
        .row_disposition = "included",
        .cap_status = "none",
        .exclusion_reason = "",
        .solve_wall_time_us = 8,
        .end_to_end_wall_time_us = 12,
        .provenance_id = "fixture",
    };
    const auto csv = argmin::bench::csv_row(row);
    require(contains(csv, "included"), "row_disposition missing from CSV row", failures);
    require(contains(csv, "none"), "cap_status missing from CSV row", failures);
    require(contains(csv, "fixture"), "provenance_id missing from CSV row", failures);
}

void check_cap_state(int& failures)
{
    contract_problem problem;
    argmin::bench::eval_counts counts;
    counts.set_max_f_evals(2);
    argmin::bench::counting_problem<contract_problem> counted{problem, counts};

    (void)counted.value(0);
    (void)counted.value(0);
    require(counts.cap_status() == "none", "f-eval cap tripped before exhaustion", failures);
    (void)counted.value(0);
    require(counts.cap_status() == "f_eval", "f-eval cap state not observable", failures);

    counts.reset();
    require(counts.cap_status() == "none", "cap state did not reset", failures);
}

void check_publication_threshold(int& failures)
{
    auto config = argmin::bench::bench_config::publication();
    require(std::abs(config.eps_feas - 1e-8) <= 0.0,
            "publication feasibility threshold default changed unexpectedly",
            failures);
    require(argmin::bench::publication_feasible(5e-9, config),
            "publication threshold rejects feasible value",
            failures);
    require(!argmin::bench::publication_feasible(5e-8, config),
            "publication threshold accepts infeasible value",
            failures);

    config.eps_feas = 1e-7;
    require(argmin::bench::publication_feasible(5e-8, config),
            "publication threshold override ignored",
            failures);
}

}

int main()
{
    int failures = 0;
    check_csv_contract(failures);
    check_cap_state(failures);
    check_publication_threshold(failures);
    return failures == 0 ? 0 : 1;
}
