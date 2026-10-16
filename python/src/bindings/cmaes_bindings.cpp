#include "bindings/registrations.h"
#include "bindings/detail/validate.h"
#include "bindings/detail/callback_set.h"
#include "bindings/detail/format_number.h"
#include "bindings/detail/wrapper_slots.h"
#include "bindings/detail/solver_methods.h"
#include "bindings/detail/solver_wrapper.h"
#include "bindings/detail/keyword_options.h"
#include "bindings/detail/problem_adapter.h"
#include "bindings/detail/describe_options.h"

#include "argmin/types.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/options/cmaes_detection_options.h"

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

using cmaes_policy_options = cmaes_policy<>::options_type;
using cmaes_restart = cmaes_policy<>::restart_strategy;
using cmaes_wrapper = solver_wrapper<cmaes_policy<>, value_problem>;

constexpr const char* cmaes_known_defect =
    "CMA-ES has no passing cell in the library's publication benchmark cohort (0 pass, 4 "
    "expected failures, 2 excluded), and it converges to local minima on the Ackley and "
    "Schwefel functions.";

void bind_cmaes_detection_options(nb::module_& m)
{
    nb::class_<cmaes_detection_options<>>(m, "CmaesDetectionOptions",
                               "Read-only view of the evolution strategy's degeneracy detection "
                               "thresholds: step-size collapse, covariance conditioning and the "
                               "two flatness tolerances.")
        .def(nb::init<>())
        .def_ro("sigma_collapse_threshold", &cmaes_detection_options<>::sigma_collapse_threshold)
        .def_ro("condition_number_limit", &cmaes_detection_options<>::condition_number_limit)
        .def_ro("objective_value_tolerance", &cmaes_detection_options<>::objective_value_tolerance)
        .def_ro("step_size_tolerance", &cmaes_detection_options<>::step_size_tolerance)
        .def("__repr__",
             [](const cmaes_detection_options<>& opts)
             {
                 return "CmaesDetectionOptions(sigma_collapse_threshold="
                        + format_number(opts.sigma_collapse_threshold)
                        + ", condition_number_limit="
                        + format_number(opts.condition_number_limit)
                        + ", objective_value_tolerance="
                        + format_number(opts.objective_value_tolerance)
                        + ", step_size_tolerance=" + format_number(opts.step_size_tolerance) + ")";
             });

    nb::enum_<cmaes_restart>(m, "CmaesRestartStrategy",
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

void bind_cmaes_solver(nb::module_& m)
{
    nb::class_<cmaes_wrapper> cls(
        m, "CmaesSolver", nb::type_slots(wrapper_slots<cmaes_wrapper>),
        "Covariance-matrix-adaptation evolution strategy. Stochastic: pass a seed to make "
        "a run reproducible, and read it back from the policy options snapshot. "
        "Experimental, with a known defect. CMA-ES has no passing cell in the library's "
        "publication benchmark cohort (0 pass, 4 expected failures, 2 excluded), and it "
        "converges to local minima on the Ackley and Schwefel functions.");
    cls.attr("maturity") = "experimental";
    cls.attr("known_defect") = cmaes_known_defect;
    cls.def(
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
        "reproducible run.");
    def_solver_methods(cls, "CmaesSolver");
}

}

void register_cmaes(nb::module_& m)
{
    bind_cmaes_detection_options(m);
    bind_cmaes_solver(m);
}

}
