// argmin variant: compile-time-only racing-tuple verification for
//                 basic_solver_group across <N, fast> + <N, accurate>
//                 mode variants of every line-search SQP policy. The
//                 racing tuple's per-policy options_type forwarding
//                 must distinguish between mode variants because
//                 kraft_slsqp_policy<N, fast>::options_type and
//                 kraft_slsqp_policy<N, accurate>::options_type are
//                 two distinct types in the std::tuple<...> backing
//                 of basic_solver_group.
//
// Reference: ARCHITECTURE.md OQ6 (basic_solver_group racing compile
//            test); KNITRO commercial-mode-system precedent.

#include "argmin/schedule/basic_solver_group.h"
#include "argmin/schedule/round_robin_schedule.h"
#include "argmin/solver/filter_nw_sqp_policy.h"
#include "argmin/solver/filter_slsqp_policy.h"
#include "argmin/solver/kraft_slsqp_policy.h"
#include "argmin/solver/tr_sqp_policy.h"
#include "argmin/solver/nw_sqp_policy.h"
#include "argmin/solver/basic_solver.h"
#include "argmin/solver/sqp_mode.h"
#include "argmin/formulation/concepts.h"
#include "argmin/types.h"

#include <type_traits>

using namespace argmin;

// -----------------------------------------------------------------------
// Gate 1 -- policy_options_t<> NTTP-forwarding alias resolution.
// -----------------------------------------------------------------------
// The two operands of the static_assert below are spelled DIFFERENTLY:
// the LHS goes through detail::policy_options_impl via the public alias
// in argmin/solver/basic_solver.h, and the RHS names the nested
// options_type directly off the mode-bearing kraft policy. They must
// resolve to the same concrete type. A regression in the alias's
// resolution path (e.g. accidentally always picking the
// solver_options<> fallback when has_options_type<Policy> is satisfied)
// would fail this check.
//
// kraft_slsqp_policy::options_type is a plain nested struct (not itself
// a template on Mode), so the LHS resolves to the
// kraft_slsqp_policy<N, fast>-nested options_type and the RHS names
// that same nested type. The check is non-trivial because it exercises
// the alias's specialization-selection logic, which `is_same_v<T, T>`
// would not.
static_assert(std::is_same_v<
    argmin::policy_options_t<
        argmin::kraft_slsqp_policy<argmin::dynamic_dimension, argmin::sqp_mode::fast>,
        double>,
    typename argmin::kraft_slsqp_policy<argmin::dynamic_dimension, argmin::sqp_mode::fast>::options_type>);
static_assert(std::is_same_v<
    argmin::policy_options_t<
        argmin::kraft_slsqp_policy<argmin::dynamic_dimension, argmin::sqp_mode::accurate>,
        double>,
    typename argmin::kraft_slsqp_policy<argmin::dynamic_dimension, argmin::sqp_mode::accurate>::options_type>);

// -----------------------------------------------------------------------
// Gate 2 -- <N, fast> != <N, accurate> distinct types across all four
// line-search SQP policies.
// -----------------------------------------------------------------------
static_assert(!std::is_same_v<
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::fast>,
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(!std::is_same_v<
    nw_sqp_policy<dynamic_dimension, sqp_mode::fast>,
    nw_sqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(!std::is_same_v<
    filter_slsqp_policy<dynamic_dimension, sqp_mode::fast>,
    filter_slsqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(!std::is_same_v<
    filter_nw_sqp_policy<dynamic_dimension, sqp_mode::fast>,
    filter_nw_sqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(!std::is_same_v<
    tr_sqp_policy<dynamic_dimension, sqp_mode::fast>,
    tr_sqp_policy<dynamic_dimension, sqp_mode::accurate>>);

// -----------------------------------------------------------------------
// Gate 3 -- _fast / _accurate alias resolution across all four policies.
// -----------------------------------------------------------------------
static_assert(std::is_same_v<
    kraft_slsqp_policy_fast<dynamic_dimension>,
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::fast>>);
static_assert(std::is_same_v<
    kraft_slsqp_policy_accurate<dynamic_dimension>,
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(std::is_same_v<
    nw_sqp_policy_fast<dynamic_dimension>,
    nw_sqp_policy<dynamic_dimension, sqp_mode::fast>>);
static_assert(std::is_same_v<
    nw_sqp_policy_accurate<dynamic_dimension>,
    nw_sqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(std::is_same_v<
    filter_slsqp_policy_fast<dynamic_dimension>,
    filter_slsqp_policy<dynamic_dimension, sqp_mode::fast>>);
static_assert(std::is_same_v<
    filter_slsqp_policy_accurate<dynamic_dimension>,
    filter_slsqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(std::is_same_v<
    filter_nw_sqp_policy_fast<dynamic_dimension>,
    filter_nw_sqp_policy<dynamic_dimension, sqp_mode::fast>>);
static_assert(std::is_same_v<
    filter_nw_sqp_policy_accurate<dynamic_dimension>,
    filter_nw_sqp_policy<dynamic_dimension, sqp_mode::accurate>>);
static_assert(std::is_same_v<
    tr_sqp_policy_fast<dynamic_dimension>,
    tr_sqp_policy<dynamic_dimension, sqp_mode::fast>>);
static_assert(std::is_same_v<
    tr_sqp_policy_accurate<dynamic_dimension>,
    tr_sqp_policy<dynamic_dimension, sqp_mode::accurate>>);

// -----------------------------------------------------------------------
// Gate 4 -- rebind<M> propagates Mode through the dimension axis.
// -----------------------------------------------------------------------
// Mode is a separate axis from N; rebind<M> must preserve Mode verbatim.
// A regression in the rebind alias (e.g. dropping Mode and silently
// defaulting to accurate) would fail this check.
static_assert(std::is_same_v<
    typename kraft_slsqp_policy<dynamic_dimension, sqp_mode::fast>::template rebind<4>,
    kraft_slsqp_policy<4, sqp_mode::fast>>);
static_assert(std::is_same_v<
    typename filter_nw_sqp_policy<dynamic_dimension, sqp_mode::fast>::template rebind<4>,
    filter_nw_sqp_policy<4, sqp_mode::fast>>);

// -----------------------------------------------------------------------
// Gate 5 -- basic_solver_group instantiation across <N, fast> + <N,
// accurate> variants of kraft_slsqp_policy.
// -----------------------------------------------------------------------
// The using-alias is the load-bearing test; its SFINAE-failure surfaces
// as a compile error if policy_options_t<> does not survive NTTP
// threading -- the per-policy options_type forwarding inside the
// std::tuple<policy_options_t<...>...> backing of basic_solver_group
// must distinguish the two variants.
namespace
{

using racing_kraft = basic_solver_group<
    round_robin_schedule,
    dynamic_dimension,
    void,
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::fast>,
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::accurate>>;

// The is_same_v<T, T> here only forces racing_kraft to be a complete
// type (the alias instantiation is the load-bearing check); without
// referencing the alias the compiler may skip alias-expansion-time
// SFINAE.
static_assert(std::is_same_v<racing_kraft, racing_kraft>);

}

// -----------------------------------------------------------------------
// nlp_solver concept satisfaction gate for tr_sqp_policy.
// -----------------------------------------------------------------------
// The trust-region SQP policy must satisfy the same nlp_solver concept
// the line-search SQP policies do, so basic_solver_group can race them
// against each other and downstream consumers (ctrlpp / cartan) can
// inject either family through the same concept-templated entry point.
static_assert(nlp_solver<basic_solver<tr_sqp_policy<dynamic_dimension, sqp_mode::accurate>>>);
static_assert(nlp_solver<basic_solver<tr_sqp_policy<dynamic_dimension, sqp_mode::fast>>>);

// -----------------------------------------------------------------------
// Gate 5b -- cross-family racing-tuple instantiation.
// -----------------------------------------------------------------------
// basic_solver_group instantiation across a line-search SQP policy
// (kraft_slsqp_accurate) and a trust-region SQP policy
// (tr_sqp_accurate). The compile-time form of the cross-family racing
// requirement: if the per-policy options_type forwarding inside the
// std::tuple<policy_options_t<...>...> backing distinguishes the two
// policy types (which have entirely different options_type layouts),
// the using-alias is well-formed and the static_assert holds; a
// regression in policy_options_t<> or in the racing-tuple machinery
// surfaces as a SFINAE failure on the alias.
namespace
{

using racing_cross_family = basic_solver_group<
    round_robin_schedule,
    dynamic_dimension,
    void,
    kraft_slsqp_policy<dynamic_dimension, sqp_mode::accurate>,
    tr_sqp_policy<dynamic_dimension, sqp_mode::accurate>>;

static_assert(std::is_same_v<racing_cross_family, racing_cross_family>);

}

int main() { return 0; }
