#include "argmin/detail/active_set_qp.h"
#include "argmin/detail/mma_subproblem.h"
#include "argmin/detail/projected_gn_step.h"
#include "argmin/detail/subspace_minimization.h"

#include "argmin/qp/dense_admm_qp.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <utility>

namespace
{

// Seeds the stack a following frame occupies, so an indeterminate enum copy has an invalid value
// for UBSan to see.
void dirty_stack()
{
    std::array<volatile std::uint8_t, 65536> junk;
    for(volatile std::uint8_t& byte : junk)
        byte = 0xAB;
}

template <typename Component>
bool move_twice(Component& built)
{
    Component moved{std::move(built)};
    Component assigned;
    assigned = std::move(moved);
    return true;
}

// An empty pack would value-initialize, zero-filling what a defaulted constructor leaves
// indeterminate; solver code default-initializes, so the test must too.
template <typename Component, typename... Args>
bool construct_and_move(Args... args)
{
    if constexpr(sizeof...(Args) == 0)
    {
        Component built;
        return move_twice(built);
    }
    else
    {
        Component built(args...);
        return move_twice(built);
    }
}

template <typename Component, typename... Args>
bool on_dirty_stack(Args... args)
{
    void (*volatile dirty)() = dirty_stack;
    bool (*volatile run)(Args...) = construct_and_move<Component, Args...>;
    dirty();
    return run(args...);
}

}

TEST_CASE("decomposition state is defined before the first factorization -- active-set QP", "[decomposition_init]")
{
    CHECK(on_dirty_stack<argmin::detail::active_set_qp_solver<double>>());
    CHECK(on_dirty_stack<argmin::detail::active_set_qp_solver<double, 4>>());
    CHECK(on_dirty_stack<argmin::detail::active_set_qp_solver<double>>(4, 3));
}

TEST_CASE("decomposition state is defined before the first factorization -- dense ADMM QP", "[decomposition_init]")
{
    CHECK(on_dirty_stack<argmin::dense_admm_qp_solver<double>>());
    CHECK(on_dirty_stack<argmin::dense_admm_qp_solver<double, 4>>());
    CHECK(on_dirty_stack<argmin::dense_admm_qp_solver<double>>(4, 3));
    CHECK(on_dirty_stack<argmin::dense_admm_qp_solver<double, 4>>(4, 3));
}

TEST_CASE("decomposition state is defined before the first factorization -- MMA subproblem", "[decomposition_init]")
{
    CHECK(on_dirty_stack<argmin::detail::mma_subproblem_solver<double>>());
    CHECK(on_dirty_stack<argmin::detail::mma_subproblem_solver<double, 3, 2>>());
    CHECK(on_dirty_stack<argmin::detail::mma_subproblem_solver<double>>(3, 0));
    CHECK(on_dirty_stack<argmin::detail::mma_subproblem_solver<double>>(3, 2));
}

TEST_CASE("decomposition state is defined before the first factorization -- bound-constrained workspaces",
          "[decomposition_init]")
{
    CHECK(on_dirty_stack<argmin::detail::subspace_minimizer<double>>());
    CHECK(on_dirty_stack<argmin::detail::subspace_minimizer<double>>(4));
    CHECK(on_dirty_stack<argmin::detail::projected_gn_workspace<double>>());
}
