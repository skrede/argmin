"""The header and struct behind each options type test_defaults_sweep parametrizes over."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Source:
    header: str
    struct: str
    minimum_fields: int
    rename: tuple = ()
    unchecked: tuple = ()

    def renamed(self, field):
        return dict(self.rename).get(field, field)

    def reason(self, field):
        return dict(self.unchecked).get(field)


@dataclasses.dataclass(frozen=True)
class Entry:
    sources: tuple


def entry(*sources):
    return Entry(sources=tuple(sources))


# Every bound options value type, mapped to the committed header and struct
# that defines it. A field listed as unchecked carries the reason it cannot be
# compared here; a field with no entry and no reason fails.
TABLE = {
    "DenseQpOptions": entry(Source("options/dense_qp_options.h", "dense_qp_options", 17)),
    "SparseQpOptions": entry(Source("options/sparse_qp_options.h", "sparse_qp_options", 17)),
    "DenseActiveSetQpOptions": entry(
        Source("options/dense_active_set_qp_options.h", "dense_active_set_qp_options", 3)
    ),
    "LineSearchOptions": entry(Source("line_search/options.h", "line_search_options", 6)),
    "QpOptions": entry(Source("options/qp_options.h", "qp_options", 2)),
    "TrustRegionOptions": entry(Source("options/trust_region_options.h", "trust_region_options", 6)),
    "CmaesDetectionOptions": entry(
        Source("options/cmaes_detection_options.h", "cmaes_detection_options", 4)
    ),
    "LbfgsbOptions": entry(Source("solver/lbfgsb_policy.h", "options_type", 3)),
    "BobyqaOptions": entry(Source("solver/bobyqa_policy.h", "options_type", 5)),
    "CobylaOptions": entry(Source("solver/cobyla_policy.h", "options_type", 4)),
    "KraftSlsqpOptions": entry(Source("solver/kraft_slsqp_policy.h", "options_type", 6)),
    "FilterSlsqpOptions": entry(Source("solver/filter_slsqp_policy.h", "options_type", 9)),
    "TrSqpOptions": entry(Source("solver/tr_sqp_policy.h", "options_type", 8)),
    "CmaesOptions": entry(
        Source(
            "solver/alternative/cmaes/pwq_reparameterization_policy.h",
            "options_type",
            9,
            rename=(("lambda", "population_size"), ("cmaes", "detection")),
            unchecked=(
                (
                    "eigendecomposition_skip_generations",
                    "not on the bound surface: an override for a formula the policy "
                    "derives, deferred with the rest of the tuning knobs",
                ),
                (
                    "stagnation_window",
                    "not on the bound surface, for the same reason as the "
                    "eigendecomposition override above",
                ),
            ),
        )
    ),
    # The driver's convergence member is a template parameter, so the header
    # that declares it carries no literal to compare. The thresholds the
    # binding exposes as flat properties live in the criterion structs, which
    # are swept here as their own sources.
    "SolverOptions": entry(
        Source(
            "solver/options.h",
            "solver_options",
            4,
            unchecked=(
                (
                    "convergence",
                    "a template parameter with no literal of its own; its criteria "
                    "are swept below as separate sources",
                ),
            ),
        ),
        Source(
            "solver/convergence.h",
            "gradient_tolerance_criterion",
            1,
            rename=(("threshold", "gradient_threshold"),),
        ),
        Source(
            "solver/convergence.h",
            "objective_tolerance_criterion",
            2,
            rename=(("threshold", "objective_threshold"),),
        ),
        Source(
            "solver/convergence.h",
            "step_tolerance_criterion",
            2,
            rename=(("threshold", "step_threshold"),),
            unchecked=(
                (
                    "feasibility_tolerance",
                    "the criterion's own copy; the driver-level tolerance of the "
                    "same name is the one the binding exposes",
                ),
            ),
        ),
        Source(
            "solver/convergence.h",
            "stall_tolerance_criterion",
            4,
            rename=(("threshold", "stall_threshold"), ("window", "stall_window")),
            unchecked=(
                (
                    "track_best_seen_feasible",
                    "not on the bound surface: a driver-internal selection switch, "
                    "not a tolerance",
                ),
                (
                    "feasibility_tolerance",
                    "the criterion's own copy, as in the step criterion above",
                ),
            ),
        ),
    ),
}

# A bound value type that is not a configuration set. Each needs a reason, so
# the escape hatch from the completeness assertion stays visible.
NOT_AN_OPTIONS_SET = {
    "QpResult": "an output value object; its fields are produced by a solve rather "
    "than configured by a caller",
    "SolveResult": "an output value object, as above",
}

STRUCT_TO_BINDING = {entry_.sources[0].struct: name for name, entry_ in TABLE.items()}
# The policy option structs are all spelled options_type, so the struct name
# alone cannot address them; nothing nests one inside another, and only the
# nested lookup uses this map.
STRUCT_TO_BINDING.pop("options_type", None)
