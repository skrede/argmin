#include "golden_file.h"
#include "family_traces.h"
#include "golden_compare.h"
#include "trajectory_trace.h"
#include "trajectory_digest.h"

#include "cases/case_options.h"

#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <algorithm>

using namespace argmin;

namespace
{

std::string flip_bit(const std::string& hex, std::int32_t bit)
{
    return golden::to_hex(static_cast<std::uint64_t>(std::stoull(hex, nullptr, 16)) ^ (std::uint64_t{1} << bit));
}

bool reported(const std::vector<std::string>& report, const std::string& line)
{
    return std::ranges::find(report, line) != report.end();
}

struct terminal_result
{
    Eigen::Vector2d x;
    double objective_value;
    solver_status status;
    std::uint32_t iterations;
    std::uint32_t function_evaluations;
};

struct terminal_solver
{
    struct state_type
    {
        Eigen::Vector2d x;
    };

    solver_status status() const { return solver_status::converged; }
    const state_type& state() const { return current; }
    step_result<double> step() { return {}; }
    terminal_result solve(const solver_options<>&) const { return terminal_result{current.x, 0.0, solver_status::converged, 0, 0}; }

    state_type current;
};

struct start_problem
{
    Eigen::Vector2d initial_point() const { return Eigen::Vector2d(1.0, -0.0); }
};

golden::traced_case final_only(double x0)
{
    golden::traced_case traced{"value|class|problem", {}, {}};
    golden::trace_final(traced, terminal_result{Eigen::Vector2d(x0, 0.0), 0.0, solver_status::converged, 0, 0});
    return traced;
}

}

TEST_CASE("trajectory comparison reports a one-bit digest flip at its case and iteration", "[trajectory-golden]")
{
    golden::trace_table live;
    golden::trace_gradient(live, golden::trace_settings{"", "", false});
    REQUIRE(live.size() > 1);
    REQUIRE(live[1].steps.size() > 3);
    golden::trace_table flipped = live;
    flipped[1].steps[3] = flip_bit(flipped[1].steps[3], 17);
    const std::vector<std::string> report = golden::compare_tables(live, flipped);
    CHECK(report == std::vector<std::string>{live[1].name + ": first diverging iteration 3"});
}

TEST_CASE("trajectory comparison tells signed zeros and NaN payloads apart", "[trajectory-golden]")
{
    const double quiet_one = std::bit_cast<double>(std::uint64_t{0x7ff8000000000001});
    const double quiet_two = std::bit_cast<double>(std::uint64_t{0x7ff8000000000002});
    CHECK(reported(golden::compare_tables({final_only(0.0)}, {final_only(-0.0)}), "value|class|problem: final x0 golden 0x0000000000000000 live 0x8000000000000000"));
    CHECK_FALSE(golden::compare_tables({final_only(quiet_one)}, {final_only(quiet_two)}).empty());
    CHECK(golden::compare_tables({final_only(quiet_one)}, {final_only(quiet_one)}).empty());
    golden::trajectory_digest positive;
    golden::trajectory_digest negative;
    positive.fold(0.0);
    negative.fold(-0.0);
    CHECK(positive.value() != negative.value());
}

TEST_CASE("a trajectory case that ends before its first step keeps its final lines", "[trajectory-golden]")
{
    golden::trace_table table;
    golden::tracing_visitor visitor(table, golden::trace_settings{"", "", true});
    const auto make_solver = [](const Eigen::Vector2d& x0) { return terminal_solver{{x0}}; };
    visitor(cases::case_id{"terminal", "unconstrained", "start"}, start_problem{}, solver_options<>{}, make_solver);
    REQUIRE(table.size() == 1);
    CHECK(table[0].steps.empty());
    CHECK(golden::final_value(table[0], "x1") == "0x8000000000000000");
    CHECK(golden::final_value(table[0], "status") == "1:converged");
    std::stringstream file;
    golden::write_golden(file, golden::running_identity(), table);
    const golden::golden_record record = golden::load_golden(file);
    CHECK(record.malformed.empty());
    CHECK(golden::compare_tables(record.table, table).empty());
}

TEST_CASE("trajectory comparison reports a case missing from either side", "[trajectory-golden]")
{
    golden::trace_table live;
    golden::trace_gradient(live, golden::trace_settings{"", "", false});
    golden::trace_table shorter = live;
    shorter.pop_back();
    CHECK(golden::compare_tables(shorter, live) == std::vector<std::string>{live.back().name + ": missing from the golden"});
    CHECK(golden::compare_tables(live, shorter) == std::vector<std::string>{live.back().name + ": missing from the live trace"});
}

TEST_CASE("a trajectory golden file reads back to the table it was written from", "[trajectory-golden]")
{
    golden::trace_table live;
    golden::trace_least_squares(live, golden::trace_settings{"", "", true});
    std::stringstream file;
    golden::write_golden(file, "identity", live);
    const golden::golden_record record = golden::load_golden(file);
    CHECK(record.identity == "identity");
    CHECK(record.malformed.empty());
    CHECK(golden::compare_tables(record.table, live).empty());
}
