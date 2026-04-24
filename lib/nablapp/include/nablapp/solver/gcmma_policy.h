#ifndef HPP_GUARD_NABLAPP_SOLVER_GCMMA_POLICY_H
#define HPP_GUARD_NABLAPP_SOLVER_GCMMA_POLICY_H

// Globally Convergent MMA (GCMMA) policy — delegates to mma_policy.
//
// With the CCSA quadratic-penalty subproblem (Svanberg 2002 / NLopt
// LD_MMA), the per-component raa machinery that distinguished GCMMA
// from MMA in the reciprocal formulation is no longer needed: the
// scalar rho / per-constraint rhoc in mma_policy already provides the
// full CCSA conservativity apparatus. gcmma_policy is therefore a thin
// wrapper that delegates to mma_policy, preserving API compatibility.
//
// References:
//   Svanberg 2002, "A class of globally convergent optimization methods
//     based on conservative convex separable approximations", SIAM J.
//     Optim. 12(2):555-573.
//   NLopt ccsa_quadratic.c (Steven G. Johnson 2008-2012).

#include "nablapp/solver/mma_policy.h"

#include <Eigen/Core>

namespace nablapp
{

template <int N = dynamic_dimension,
          template<int> typename DualPolicy = lbfgsb_policy>
struct gcmma_policy
{
    using scalar_type = double;

    template <int M>
    using rebind = gcmma_policy<M, DualPolicy>;

    struct options_type
    {
        typename mma_policy<N, DualPolicy>::options_type mma_opts{};

        std::optional<std::uint16_t> max_inner_iterations{};
        std::optional<double> stall_tolerance_threshold{};
        mma_subproblem_options subproblem{};
        std::uint16_t stall_window{50};
    };

    template <typename P = void>
    struct state_type
    {
        static constexpr int M = [] {
            if constexpr(has_constraint_count<P>) return constraint_count_v<P>;
            else return dynamic_dimension;
        }();

        typename mma_policy<N, DualPolicy>::template state_type<P> mma_state;
        options_type opts;

        // Proxy members for basic_solver compatibility.
        Eigen::Vector<double, N>& x = mma_state.x;
        Eigen::VectorXd& c_eq = mma_state.c_eq;
        Eigen::Vector<double, M>& c_ineq = mma_state.c_ineq;

        state_type() = default;
        state_type(const state_type& o)
            : mma_state{o.mma_state}, opts{o.opts}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type(state_type&& o) noexcept
            : mma_state{std::move(o.mma_state)}, opts{std::move(o.opts)}
            , x{mma_state.x}, c_eq{mma_state.c_eq}, c_ineq{mma_state.c_ineq}
        {}
        state_type& operator=(const state_type& o)
        {
            if(this != &o)
            {
                mma_state = o.mma_state;
                opts = o.opts;
            }
            return *this;
        }
        state_type& operator=(state_type&& o) noexcept
        {
            if(this != &o)
            {
                mma_state = std::move(o.mma_state);
                opts = std::move(o.opts);
            }
            return *this;
        }
    };

    template <typename Problem, typename Convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& sopts,
                             const options_type& policy_opts)
    {
        state_type<Problem> s;
        s.opts = policy_opts;
        // Forward max_inner and subproblem options into mma_opts.
        auto mma_o = policy_opts.mma_opts;
        if(policy_opts.max_inner_iterations)
            mma_o.max_inner_iterations = policy_opts.max_inner_iterations;
        if(policy_opts.stall_tolerance_threshold)
            mma_o.stall_tolerance_threshold = policy_opts.stall_tolerance_threshold;
        mma_o.stall_window = policy_opts.stall_window;
        mma_o.subproblem = policy_opts.subproblem;
        s.mma_state = mma_policy<N, DualPolicy>{}.init(problem, x0, sopts, mma_o);
        return s;
    }

    template <typename Problem, typename Convergence = default_convergence>
    state_type<Problem> init(const Problem& problem,
                             const Eigen::Vector<double, N>& x0,
                             const solver_options<Convergence>& sopts)
    {
        state_type<Problem> s;
        s.mma_state = mma_policy<N, DualPolicy>{}.init(problem, x0, sopts);
        return s;
    }

    template <typename P>
    step_result<double> step(state_type<P>& s)
    {
        return mma_policy<N, DualPolicy>{}.step(s.mma_state);
    }

    template <typename P>
    void reset(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        mma_policy<N, DualPolicy>{}.reset(s.mma_state, x0);
    }

    template <typename P>
    void reset_clear(state_type<P>& s, const Eigen::Vector<double, N>& x0)
    {
        mma_policy<N, DualPolicy>{}.reset_clear(s.mma_state, x0);
    }
};

}

#endif
