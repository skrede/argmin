#include "bindings/detail/errors.h"
#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/solver/bobyqa_policy.h"
#include "argmin/solver/cobyla_policy.h"
#include "argmin/options/cmaes_options.h"
#include "argmin/options/trust_region_options.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

#include <new>
#include <memory>
#include <string>
#include <cstdint>
#include <utility>
#include <optional>

namespace nb = nanobind;

namespace argmin::python
{

namespace
{

using bobyqa_options = bobyqa_policy<>::options_type;
using cobyla_options = cobyla_policy::options_type;
using cmaes_policy_options = cmaes_policy<>::options_type;
using cmaes_restart = cmaes_policy<>::restart_strategy;

using bobyqa_wrapper = solver_wrapper<bobyqa_policy<>, bounded_problem>;
using cobyla_wrapper = solver_wrapper<cobyla_policy, constrained_values_problem>;
using cmaes_wrapper = solver_wrapper<cmaes_policy<>, value_problem>;

std::string describe_optional(const std::optional<double>& value)
{
    return value ? format_number(*value) : std::string("None");
}

void bind_trust_region_options(nb::module_& m)
{
    nb::class_<trust_region_options>(m, "TrustRegionOptions",
                                    "Read-only view of the trust-region acceptance and radius "
                                    "update parameters.")
        .def(nb::init<>())
        .def_ro("eta_good", &trust_region_options::eta_good)
        .def_ro("eta_poor", &trust_region_options::eta_poor)
        .def_ro("expand_factor", &trust_region_options::expand_factor)
        .def_ro("shrink_factor", &trust_region_options::shrink_factor)
        .def_ro("step_threshold", &trust_region_options::step_threshold)
        .def_ro("geometry_factor", &trust_region_options::geometry_factor)
        .def("__repr__",
             [](const trust_region_options& opts)
             {
                 return "TrustRegionOptions(eta_good=" + format_number(opts.eta_good)
                        + ", eta_poor=" + format_number(opts.eta_poor)
                        + ", expand_factor=" + format_number(opts.expand_factor)
                        + ", shrink_factor=" + format_number(opts.shrink_factor)
                        + ", step_threshold=" + format_number(opts.step_threshold)
                        + ", geometry_factor=" + format_number(opts.geometry_factor) + ")";
             });
}

void bind_bobyqa_options(nb::module_& m)
{
    nb::class_<bobyqa_options>(m, "BobyqaOptions",
                              "Read-only view of the bound-constrained interpolation "
                              "policy's own configuration.")
        .def(nb::init<>())
        .def_ro("initial_trust_radius", &bobyqa_options::initial_trust_radius)
        .def_ro("final_trust_radius", &bobyqa_options::final_trust_radius)
        .def_ro("trust", &bobyqa_options::trust)
        .def_ro("stall_window", &bobyqa_options::stall_window)
        .def_ro("feasibility_gate", &bobyqa_options::feasibility_gate)
        .def("__repr__",
             [](const bobyqa_options& opts)
             {
                 return "BobyqaOptions(initial_trust_radius="
                        + describe_optional(opts.initial_trust_radius) + ", final_trust_radius="
                        + describe_optional(opts.final_trust_radius)
                        + ", stall_window=" + format_number(opts.stall_window)
                        + ", feasibility_gate=" + format_number(opts.feasibility_gate) + ")";
             });
}

void bind_cobyla_options(nb::module_& m)
{
    nb::class_<cobyla_options>(m, "CobylaOptions",
                              "Read-only view of the constrained linear-approximation "
                              "policy's own configuration.")
        .def(nb::init<>())
        .def_ro("initial_trust_radius", &cobyla_options::initial_trust_radius)
        .def_ro("final_trust_radius", &cobyla_options::final_trust_radius)
        .def_ro("stall_window", &cobyla_options::stall_window)
        .def_ro("feasibility_gate", &cobyla_options::feasibility_gate)
        .def("__repr__",
             [](const cobyla_options& opts)
             {
                 return "CobylaOptions(initial_trust_radius="
                        + format_number(opts.initial_trust_radius)
                        + ", final_trust_radius=" + format_number(opts.final_trust_radius)
                        + ", stall_window=" + format_number(opts.stall_window)
                        + ", feasibility_gate=" + format_number(opts.feasibility_gate) + ")";
             });
}

void bind_cmaes_options(nb::module_& m)
{
    nb::class_<cmaes_options>(m, "CmaesDetectionOptions",
                             "Read-only view of the evolution strategy's degeneracy detection "
                             "thresholds: step-size collapse, covariance conditioning and the "
                             "two flatness tolerances.")
        .def(nb::init<>())
        .def_ro("sigma_collapse_threshold", &cmaes_options::sigma_collapse_threshold)
        .def_ro("condition_number_limit", &cmaes_options::condition_number_limit)
        .def_ro("objective_value_tolerance", &cmaes_options::objective_value_tolerance)
        .def_ro("step_size_tolerance", &cmaes_options::step_size_tolerance)
        .def("__repr__",
             [](const cmaes_options& opts)
             {
                 return "CmaesDetectionOptions(sigma_collapse_threshold="
                        + format_number(opts.sigma_collapse_threshold)
                        + ", condition_number_limit="
                        + format_number(opts.condition_number_limit)
                        + ", objective_value_tolerance="
                        + format_number(opts.objective_value_tolerance)
                        + ", step_size_tolerance=" + format_number(opts.step_size_tolerance) + ")";
             });

    nb::enum_<cmaes_restart>(m, "CmaesRestart",
                            "Whether the evolution strategy restarts with an increasing "
                            "population after an exit criterion fires.")
        .value("none", cmaes_restart::none)
        .value("ipop", cmaes_restart::ipop);

    nb::class_<cmaes_policy_options>(m, "CmaesOptions",
                                    "Read-only view of the covariance-adaptation evolution "
                                    "strategy's own configuration, including the seed that "
                                    "makes a run reproducible.")
        .def(nb::init<>())
        .def_ro("population_size", &cmaes_policy_options::lambda)
        .def_ro("initial_sigma", &cmaes_policy_options::initial_sigma)
        .def_ro("restart", &cmaes_policy_options::restart)
        .def_ro("seed", &cmaes_policy_options::seed)
        .def_ro("detection", &cmaes_policy_options::cmaes)
        .def_ro("stall_window", &cmaes_policy_options::stall_window)
        .def_ro("feasibility_gate", &cmaes_policy_options::feasibility_gate)
        .def("__repr__",
             [](const cmaes_policy_options& opts)
             {
                 const std::string population =
                     opts.lambda ? format_number(*opts.lambda) : std::string("None");
                 const std::string seed =
                     opts.seed ? format_number(*opts.seed) : std::string("None");
                 return "CmaesOptions(population_size=" + population + ", initial_sigma="
                        + describe_optional(opts.initial_sigma) + ", seed=" + seed
                        + ", stall_window=" + format_number(opts.stall_window) + ")";
             });
}

void bind_bobyqa_solver(nb::module_& m)
{
    nb::class_<bobyqa_wrapper>(
        m, "BobyqaSolver", nb::type_slots(wrapper_slots<bobyqa_wrapper>),
        "Bound-constrained derivative-free trust-region interpolation method. Takes an "
        "objective only; no gradient is used or requested.")
        .def(
            "__init__",
            [](bobyqa_wrapper* self, nb::object objective, const vector<double>& x0,
               nb::object lower_bounds, nb::object upper_bounds,
               std::optional<double> initial_trust_radius,
               std::optional<double> final_trust_radius, std::optional<int> policy_stall_window,
               std::optional<int> max_iterations, std::optional<double> feasibility_tolerance,
               std::optional<double> constraint_tolerance,
               std::optional<double> gradient_threshold,
               std::optional<double> objective_threshold, std::optional<double> step_threshold,
               std::optional<double> stall_threshold, std::optional<int> stall_window)
            {
                check_positive_dimension(static_cast<int>(x0.size()), "x0");
                check_all_finite(x0, "x0");
                require_callable(objective, "objective");

                const solver_options<> opts =
                    configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                               gradient_threshold, objective_threshold, step_threshold,
                               stall_threshold, stall_window});

                bobyqa_options policy_opts;
                if(initial_trust_radius)
                    policy_opts.initial_trust_radius =
                        positive_option(*initial_trust_radius, "initial_trust_radius");
                if(final_trust_radius)
                    policy_opts.final_trust_radius =
                        positive_option(*final_trust_radius, "final_trust_radius");
                if(policy_stall_window)
                {
                    check_positive_dimension(*policy_stall_window, "policy_stall_window");
                    policy_opts.stall_window =
                        static_cast<std::uint16_t>(*policy_stall_window);
                }

                callback_set calls;
                calls.objective = std::move(objective);
                calls.lower_bounds = std::move(lower_bounds);
                calls.upper_bounds = std::move(upper_bounds);

                auto adapter = std::make_unique<bounded_problem>(static_cast<int>(x0.size()),
                                                                 std::move(calls));
                new(self) bobyqa_wrapper(std::move(adapter), x0, opts, policy_opts);
            },
            nb::arg("objective"), nb::arg("x0"), nb::arg("lower_bounds") = nb::none(),
            nb::arg("upper_bounds") = nb::none(), nb::arg("initial_trust_radius") = nb::none(),
            nb::arg("final_trust_radius") = nb::none(),
            nb::arg("policy_stall_window") = nb::none(), nb::arg("max_iterations") = nb::none(),
            nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(),
            nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
            nb::arg("stall_window") = nb::none(),
            "Powell's bound-constrained trust-region method over a quadratic interpolation "
            "model, requiring objective values only (Powell 2009, DAMTP 2009/NA06).")
        .def("solve", &bobyqa_wrapper::solve)
        .def("step", &bobyqa_wrapper::step)
        .def("step_n", &bobyqa_wrapper::step_n, nb::arg("budget"))
        .def("reset", &bobyqa_wrapper::reset, nb::arg("x0"))
        .def("reset_clear", &bobyqa_wrapper::reset_clear, nb::arg("x0"))
        .def("abort", &bobyqa_wrapper::abort)
        .def("status", &bobyqa_wrapper::status)
        .def("options", &bobyqa_wrapper::options)
        .def("policy_options", &bobyqa_wrapper::policy_options)
        .def_prop_ro("x", &bobyqa_wrapper::x)
        .def_prop_ro("n", &bobyqa_wrapper::dimension)
        .def_prop_ro("gradient_norm", &bobyqa_wrapper::gradient_norm)
        .def_prop_ro("constraint_violation", &bobyqa_wrapper::constraint_violation)
        .def("__repr__",
             [](const bobyqa_wrapper& solver)
             { return "BobyqaSolver(n=" + format_number(solver.dimension()) + ")"; });
}

void bind_cobyla_solver(nb::module_& m)
{
    nb::class_<cobyla_wrapper>(
        m, "CobylaSolver", nb::type_slots(wrapper_slots<cobyla_wrapper>),
        "Derivative-free constrained optimization by linear approximations. Takes an "
        "objective, a constraint vector and the two constraint counts; box bounds are "
        "rewritten into explicit constraints by the method itself.")
        .def(
            "__init__",
            [](cobyla_wrapper* self, nb::object objective, const vector<double>& x0,
               nb::object constraints, std::optional<int> num_equality,
               std::optional<int> num_inequality, nb::object lower_bounds,
               nb::object upper_bounds, std::optional<double> initial_trust_radius,
               std::optional<double> final_trust_radius, std::optional<int> policy_stall_window,
               std::optional<int> max_iterations, std::optional<double> feasibility_tolerance,
               std::optional<double> constraint_tolerance,
               std::optional<double> gradient_threshold,
               std::optional<double> objective_threshold, std::optional<double> step_threshold,
               std::optional<double> stall_threshold, std::optional<int> stall_window)
            {
                check_positive_dimension(static_cast<int>(x0.size()), "x0");
                check_all_finite(x0, "x0");
                require_callable(objective, "objective");
                require_callable(constraints, "constraints");

                const solver_options<> opts =
                    configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                               gradient_threshold, objective_threshold, step_threshold,
                               stall_threshold, stall_window});

                cobyla_options policy_opts;
                if(initial_trust_radius)
                    policy_opts.initial_trust_radius =
                        positive_option(*initial_trust_radius, "initial_trust_radius");
                if(final_trust_radius)
                    policy_opts.final_trust_radius =
                        positive_option(*final_trust_radius, "final_trust_radius");
                if(policy_stall_window)
                {
                    check_positive_dimension(*policy_stall_window, "policy_stall_window");
                    policy_opts.stall_window =
                        static_cast<std::uint16_t>(*policy_stall_window);
                }

                callback_set calls;
                calls.objective = std::move(objective);
                calls.constraints = std::move(constraints);
                calls.lower_bounds = std::move(lower_bounds);
                calls.upper_bounds = std::move(upper_bounds);
                calls.num_equality = constraint_count(num_equality, "num_equality");
                calls.num_inequality = constraint_count(num_inequality, "num_inequality");
                check_positive_dimension(calls.num_equality + calls.num_inequality,
                                         "num_equality + num_inequality");

                auto adapter = std::make_unique<constrained_values_problem>(
                    static_cast<int>(x0.size()), std::move(calls));
                new(self) cobyla_wrapper(std::move(adapter), x0, opts, policy_opts);
            },
            nb::arg("objective"), nb::arg("x0"), nb::arg("constraints"),
            nb::arg("num_equality") = nb::none(), nb::arg("num_inequality") = nb::none(),
            nb::arg("lower_bounds") = nb::none(), nb::arg("upper_bounds") = nb::none(),
            nb::arg("initial_trust_radius") = nb::none(),
            nb::arg("final_trust_radius") = nb::none(),
            nb::arg("policy_stall_window") = nb::none(), nb::arg("max_iterations") = nb::none(),
            nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(),
            nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
            nb::arg("stall_window") = nb::none(),
            "Powell's constrained trust-region method over linear interpolation models of the "
            "objective and every constraint, requiring no derivatives (Powell 1994). "
            "Equalities come first in the constraint vector and are driven to zero; "
            "inequalities follow and are held non-negative.")
        .def("solve", &cobyla_wrapper::solve)
        .def("step", &cobyla_wrapper::step)
        .def("step_n", &cobyla_wrapper::step_n, nb::arg("budget"))
        .def("reset", &cobyla_wrapper::reset, nb::arg("x0"))
        .def("reset_clear", &cobyla_wrapper::reset_clear, nb::arg("x0"))
        .def("abort", &cobyla_wrapper::abort)
        .def("status", &cobyla_wrapper::status)
        .def("options", &cobyla_wrapper::options)
        .def("policy_options", &cobyla_wrapper::policy_options)
        .def_prop_ro("x", &cobyla_wrapper::x)
        .def_prop_ro("n", &cobyla_wrapper::dimension)
        .def_prop_ro("gradient_norm", &cobyla_wrapper::gradient_norm)
        .def_prop_ro("constraint_violation", &cobyla_wrapper::constraint_violation)
        .def("__repr__",
             [](const cobyla_wrapper& solver)
             { return "CobylaSolver(n=" + format_number(solver.dimension()) + ")"; });
}

void bind_cmaes_solver(nb::module_& m)
{
    nb::class_<cmaes_wrapper>(
        m, "CmaesSolver", nb::type_slots(wrapper_slots<cmaes_wrapper>),
        "Covariance-matrix-adaptation evolution strategy. Stochastic: pass a seed to make "
        "a run reproducible, and read it back from the policy options snapshot.")
        .def(
            "__init__",
            [](cmaes_wrapper* self, nb::object objective, const vector<double>& x0,
               std::optional<int> population_size, std::optional<double> initial_sigma,
               std::optional<std::uint64_t> seed, std::optional<int> policy_stall_window,
               std::optional<int> max_iterations, std::optional<double> feasibility_tolerance,
               std::optional<double> constraint_tolerance,
               std::optional<double> gradient_threshold,
               std::optional<double> objective_threshold, std::optional<double> step_threshold,
               std::optional<double> stall_threshold, std::optional<int> stall_window)
            {
                check_positive_dimension(static_cast<int>(x0.size()), "x0");
                check_all_finite(x0, "x0");
                require_callable(objective, "objective");

                const solver_options<> opts =
                    configure({max_iterations, feasibility_tolerance, constraint_tolerance,
                               gradient_threshold, objective_threshold, step_threshold,
                               stall_threshold, stall_window});

                cmaes_policy_options policy_opts;
                if(population_size)
                {
                    check_positive_dimension(*population_size, "population_size");
                    policy_opts.lambda = static_cast<std::uint32_t>(*population_size);
                }
                if(initial_sigma)
                    policy_opts.initial_sigma = positive_option(*initial_sigma, "initial_sigma");
                if(seed)
                    policy_opts.seed = *seed;
                if(policy_stall_window)
                {
                    check_positive_dimension(*policy_stall_window, "policy_stall_window");
                    policy_opts.stall_window =
                        static_cast<std::uint16_t>(*policy_stall_window);
                }

                callback_set calls;
                calls.objective = std::move(objective);

                auto adapter = std::make_unique<value_problem>(static_cast<int>(x0.size()),
                                                               std::move(calls));
                new(self) cmaes_wrapper(std::move(adapter), x0, opts, policy_opts);
            },
            nb::arg("objective"), nb::arg("x0"), nb::arg("population_size") = nb::none(),
            nb::arg("initial_sigma") = nb::none(), nb::arg("seed") = nb::none(),
            nb::arg("policy_stall_window") = nb::none(), nb::arg("max_iterations") = nb::none(),
            nb::arg("feasibility_tolerance") = nb::none(),
            nb::arg("constraint_tolerance") = nb::none(),
            nb::arg("gradient_threshold") = nb::none(),
            nb::arg("objective_threshold") = nb::none(),
            nb::arg("step_threshold") = nb::none(), nb::arg("stall_threshold") = nb::none(),
            nb::arg("stall_window") = nb::none(),
            "Covariance matrix adaptation evolution strategy with the piecewise-quadratic "
            "bound reparameterization (Hansen 2023, arXiv:1604.00772). Supply seed for a "
            "reproducible run.")
        .def("solve", &cmaes_wrapper::solve)
        .def("step", &cmaes_wrapper::step)
        .def("step_n", &cmaes_wrapper::step_n, nb::arg("budget"))
        .def("reset", &cmaes_wrapper::reset, nb::arg("x0"))
        .def("reset_clear", &cmaes_wrapper::reset_clear, nb::arg("x0"))
        .def("abort", &cmaes_wrapper::abort)
        .def("status", &cmaes_wrapper::status)
        .def("options", &cmaes_wrapper::options)
        .def("policy_options", &cmaes_wrapper::policy_options)
        .def_prop_ro("x", &cmaes_wrapper::x)
        .def_prop_ro("n", &cmaes_wrapper::dimension)
        .def_prop_ro("gradient_norm", &cmaes_wrapper::gradient_norm)
        .def_prop_ro("constraint_violation", &cmaes_wrapper::constraint_violation)
        .def("__repr__",
             [](const cmaes_wrapper& solver)
             { return "CmaesSolver(n=" + format_number(solver.dimension()) + ")"; });
}

}

void register_nlp_derivative_free(nb::module_& m)
{
    bind_trust_region_options(m);
    bind_bobyqa_options(m);
    bind_cobyla_options(m);
    bind_cmaes_options(m);
    bind_bobyqa_solver(m);
    bind_cobyla_solver(m);
    bind_cmaes_solver(m);
}

}
