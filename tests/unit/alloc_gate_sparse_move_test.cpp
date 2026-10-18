#include "alloc_gate_support.h"

#include "sparse_admm_move_support.h"

#include <optional>

namespace
{

using argmin::alloc_gate::run_armed;
using argmin::alloc_gate::window_counts;
using argmin::control_qp_data::control_qp;
using argmin::control_qp_data::make_control_qp;
using namespace argmin::sparse_admm_move;
using solver_type = argmin::sparse_admm_qp_solver<double>;

void pose_with_polish(solver_type& solver, const control_qp& cell, const std::vector<step_problem>& schedule)
{
    const result_list run = drive(solver, cell, slice(schedule, 0, 6), options_for(true, true), true);
    REQUIRE(all_steps_solved(run));
    REQUIRE(any_polished(run));
    REQUIRE(solver.polish_analyses() > 0);
}

void check_window(const char* window, const window_counts& counts)
{
    INFO(window);
    CHECK(counts.eigen == 0);
    CHECK(counts.c == 0);
}

void check_after_moves(solver_type& assigned, solver_type& moved_from, const step_problem& next,
                       const argmin::sparse_qp_options<>& opts)
{
    argmin::qp_result<double> out;
    REQUIRE_FALSE(assigned.resolve_into(next.q, next.l, next.u, out, opts).has_value());
    CHECK(out.iterations > 0);
    const std::optional<argmin::qp_error> refused = moved_from.resolve_into(next.q, next.l, next.u, out, opts);
    REQUIRE(refused.has_value());
    CHECK(*refused == argmin::qp_error::invalid_problem);
}

}

// Sensor A cannot see sparse storage (CompressedStorage uses new[], SparseMatrix uses std::malloc),
// so the C counter is the witness, live on Linux only.
TEST_CASE("qp_sparse_admm_move_allocation_probe", "[qp-move-alloc]")
{
    const control_qp cell = make_control_qp(10, 4, 2, false, 1000003u);
    const control_qp other = make_control_qp(10, 6, 3, true, 1000033u);
    const std::vector<step_problem> schedule = receding_schedule(cell, 7);
    const argmin::sparse_qp_options<> opts = options_for(true, true);

    solver_type S;
    pose_with_polish(S, cell, schedule);
    solver_type T;
    REQUIRE(all_steps_solved(drive(T, other, receding_schedule(other, 1), opts, true)));
    solver_type U;
    std::optional<solver_type> moved;
    std::optional<solver_type> moved_unposed;

    check_window("move-construct posed", run_armed([&] { moved.emplace(std::move(S)); }));
    check_window("move-assign onto posed", run_armed([&] { T = std::move(*moved); }));
    check_window("move-construct never posed", run_armed([&] { moved_unposed.emplace(std::move(U)); }));
    check_after_moves(T, *moved, schedule[6], opts);
}
