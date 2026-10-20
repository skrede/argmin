#include "sparse_admm_move_support.h"

#include "argmin/qp/detail/sparse_kkt.h"

#include "argmin/qp/sparse_admm_qp.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <utility>
#include <optional>
#include <type_traits>

using argmin::sparse_admm_qp_solver;
using argmin::control_qp_data::control_qp;
using argmin::control_qp_data::make_control_qp;
using namespace argmin::sparse_admm_move;

static_assert(std::is_nothrow_move_constructible_v<sparse_admm_qp_solver<double>>);
static_assert(std::is_nothrow_move_assignable_v<sparse_admm_qp_solver<double>>);
static_assert(!std::is_copy_constructible_v<sparse_admm_qp_solver<double>>);
static_assert(!std::is_copy_assignable_v<sparse_admm_qp_solver<double>>);
static_assert(std::is_nothrow_move_constructible_v<argmin::detail::sparse_kkt<double>>);

namespace
{

constexpr std::int32_t steps = 16;

struct fixture
{
    control_qp cell;
    std::vector<step_problem> schedule;
    std::vector<step_problem> head;
    std::vector<step_problem> tail;
    argmin::sparse_qp_options<> opts;
};

fixture make_fixture(bool polish, bool warm)
{
    const control_qp cell = make_control_qp(10, 4, 2, false, 1000003u);
    std::vector<step_problem> schedule = receding_schedule(cell, steps);
    std::vector<step_problem> head = slice(schedule, 0, 1);
    std::vector<step_problem> tail = slice(schedule, 1, schedule.size());
    return {cell, std::move(schedule), std::move(head), std::move(tail), options_for(polish, warm)};
}

void require_twins(const result_list& a, const result_list& b)
{
    REQUIRE(a.size() == b.size());
    for(std::size_t k = 0; k < a.size(); ++k)
    {
        INFO("step " << k);
        CHECK(bit_identical(a[k], b[k]));
    }
}

void check_path_entry(const result_list& run, bool polish, bool warm)
{
    CHECK(all_steps_solved(run));
    if(polish)
        CHECK(any_polished(run));
    if(warm)
        CHECK(warm_after_first(run));
}

}

TEST_CASE("sparse_admm_qp move -- a move-constructed posed solver resolves bit-identically", "[qp][sparse_admm_move]")
{
    const bool polish = GENERATE(false, true);
    const bool warm = GENERATE(false, true);
    INFO("polish " << polish << " warm " << warm);
    const fixture f = make_fixture(polish, warm);

    sparse_admm_qp_solver<double> A;
    sparse_admm_qp_solver<double> B;
    result_list run_a = drive(A, f.cell, f.head, f.opts, true);
    result_list run_b = drive(B, f.cell, f.head, f.opts, true);
    sparse_admm_qp_solver<double> C(std::move(B));
    append(run_a, drive(A, f.cell, f.tail, f.opts, false));
    append(run_b, drive(C, f.cell, f.tail, f.opts, false));

    check_path_entry(run_a, polish, warm);
    require_twins(run_a, run_b);
}

TEST_CASE("sparse_admm_qp move -- a move-assigned posed solver resolves bit-identically", "[qp][sparse_admm_move]")
{
    const bool polish = GENERATE(false, true);
    const bool warm = GENERATE(false, true);
    INFO("polish " << polish << " warm " << warm);
    const fixture f = make_fixture(polish, warm);
    const control_qp other = make_control_qp(10, 6, 3, true, 1000033u);

    sparse_admm_qp_solver<double> A;
    sparse_admm_qp_solver<double> B;
    sparse_admm_qp_solver<double> D;
    REQUIRE(all_steps_solved(drive(D, other, receding_schedule(other, 1), f.opts, true)));
    result_list run_a = drive(A, f.cell, f.head, f.opts, true);
    result_list run_b = drive(B, f.cell, f.head, f.opts, true);
    D = std::move(B);
    append(run_a, drive(A, f.cell, f.tail, f.opts, false));
    append(run_b, drive(D, f.cell, f.tail, f.opts, false));

    check_path_entry(run_a, polish, warm);
    require_twins(run_a, run_b);
}

TEST_CASE("sparse_admm_qp move -- a solver moved before posing solves bit-identically", "[qp][sparse_admm_move]")
{
    const bool polish = GENERATE(false, true);
    const bool warm = GENERATE(false, true);
    INFO("polish " << polish << " warm " << warm);
    const fixture f = make_fixture(polish, warm);

    sparse_admm_qp_solver<double> fresh;
    sparse_admm_qp_solver<double> unused;
    sparse_admm_qp_solver<double> moved(std::move(unused));
    const result_list run_fresh = drive(fresh, f.cell, f.schedule, f.opts, true);
    const result_list run_moved = drive(moved, f.cell, f.schedule, f.opts, true);

    check_path_entry(run_fresh, polish, warm);
    require_twins(run_fresh, run_moved);
}

TEST_CASE("sparse_admm_qp move -- a moved-from solver refuses resolve and re-poses from scratch", "[qp][sparse_admm_move]")
{
    const bool polish = GENERATE(false, true);
    const bool warm = GENERATE(false, true);
    INFO("polish " << polish << " warm " << warm);
    const fixture f = make_fixture(polish, warm);

    sparse_admm_qp_solver<double> B;
    REQUIRE(all_steps_solved(drive(B, f.cell, f.schedule, f.opts, true)));
    sparse_admm_qp_solver<double> C(std::move(B));

    argmin::qp_result<double> out;
    const step_problem& s = f.schedule[1];
    const std::optional<argmin::qp_error> refused = B.resolve_into(s.q, s.l, s.u, out, f.opts);
    REQUIRE(refused.has_value());
    CHECK(*refused == argmin::qp_error::invalid_problem);

    sparse_admm_qp_solver<double> fresh;
    const result_list run_fresh = drive(fresh, f.cell, f.schedule, f.opts, true);
    const result_list run_b = drive(B, f.cell, f.schedule, f.opts, true);

    check_path_entry(run_fresh, polish, warm);
    require_twins(run_fresh, run_b);
}
