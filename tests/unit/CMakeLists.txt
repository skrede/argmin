include(Catch)

add_executable(argmin_unit_tests
    detail_test.cpp
    lm_test.cpp
    sqp_test.cpp
    ccsa_quadratic_test.cpp
    auglag_mma_instantiation_test.cpp
    auglag_gcmma_equality_test.cpp
    mma_equality_reject_test.cpp
    mma_reciprocal_test.cpp
    gcmma_move_limit_shrink_test.cpp
    gcmma_rho_wval_test.cpp
    gcmma_raa_augmented_test.cpp
    isres_test.cpp
    isres_alternatives_test.cpp
    isres_determinism_test.cpp
    cobyla_test.cpp
    bobyqa_test.cpp
    bobyqa_bounds_test.cpp
    bobyqa_xbase_shift_test.cpp
    lbfgsb_test.cpp
    cmaes_test.cpp
    cmaes_invalid_problem_test.cpp
    cmaes_sampling_test.cpp
    restarting_policy_test.cpp
    tolerance_test.cpp
    budget_driver_test.cpp
    runner_options_matrix_test.cpp
    robustness/abort_test.cpp
    robustness/restoration_terminal_test.cpp
    robustness/filter_restoration_test.cpp
    robustness/multistart_restart_test.cpp
    robustness/mma_subproblem_dual_test.cpp
    robustness/kraft_recovery_test.cpp
    scheduling_test.cpp
    convergence_test.cpp
    line_search_test.cpp
    multistart_test.cpp
    projected_gn_test.cpp
    solver_group_test.cpp
    kraft_slsqp_test.cpp
    kraft_lsq_qp_test.cpp
    kraft_lsq_qp_recovery_test.cpp
    dense_admm_qp_test.cpp
    sparse_admm_qp_test.cpp
    maros_meszaros_dense_test.cpp
    step_budget_solver_test.cpp
    stepper_test.cpp
    active_set_qp_test.cpp
    lsei_mode4_test.cpp
    compact_lbfgs_test.cpp
    test_functions_test.cpp
    filter_nw_sqp_test.cpp
    hock_schittkowski_test.cpp
    augmented_lagrangian_test.cpp
    convergence_policy_test.cpp
    filter_sqp_test.cpp
    zero_copy_test.cpp
    finite_difference_test.cpp
    kkt_residual_test.cpp
    sqp_common_joint_test.cpp
    soc_seed_projection_test.cpp
    dense_ldl_bfgs_test.cpp
    sqp_step_result_consistency_test.cpp
    steihaug_cg_test.cpp
    byrd_omojokun_test.cpp
    cauchy_point_test.cpp
    restoration_test.cpp
    tr_sqp_test.cpp
    filter_trsqp_test.cpp
    nmpc_lqr_test.cpp
    ik_pose_batch_test.cpp
    policy_move_asan_test.cpp
    filter_acceptance_test.cpp
    interpolation_delta_property_test.cpp
    hs_gradient_fd_test.cpp
    cmaes_generation_pin_test.cpp
    cobyla_parmu_trajectory_test.cpp
    cmaes_reset_parity_test.cpp
    cobyla_determinism_test.cpp
    leaf_cap_binding_test.cpp
    isres_best_track_test.cpp
    halton_distinctness_test.cpp
    mma_coefficient_pin_test.cpp
    solver_bit_identity_pin_test.cpp
    options_test.cpp
    slsqp_line_search_witness_test.cpp
    nw_sqp_soc_witness_test.cpp
    kraft_sigma_bump_witness_test.cpp
    merit_function_test.cpp
    substrate_scale_witness_test.cpp
    sparse_kkt_witness_test.cpp
    expected_test.cpp
)

target_link_libraries(argmin_unit_tests
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)

target_compile_options(argmin_unit_tests PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})
if(ARGMIN_LTO_SUPPORTED)
    set_target_properties(argmin_unit_tests PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

catch_discover_tests(argmin_unit_tests)

# Warm-start reset boundary sensor. Compiled into its own executable with
# EIGEN_RUNTIME_NO_MALLOC so the macro's Eigen-side allocation guards are
# scoped to this translation unit (defining it project-wide would perturb every
# other unit's Eigen inlining). The test arms Eigen::internal's malloc gate
# around a warm reset and asserts no Eigen heap allocation occurs at fixed or
# dynamic N; it also pins the filter_slsqp re-seed and projected_gn tau fixes.
add_executable(argmin_warm_start_boundary_test
    warm_start_boundary_test.cpp
)
target_link_libraries(argmin_warm_start_boundary_test
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)
target_compile_definitions(argmin_warm_start_boundary_test
    PRIVATE EIGEN_RUNTIME_NO_MALLOC)
target_compile_options(argmin_warm_start_boundary_test
    PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})
catch_discover_tests(argmin_warm_start_boundary_test)

# Zero-allocation correctness gates for the real-time-claimed policies. One
# dedicated executable holds every gate as its own TEST_CASE; catch_discover_tests
# exposes each as an independent ctest entry. The target defines BOTH
# ARGMIN_BENCH_TRACE_ALLOC=1 (arming the diagnostics header's LIVE sensor branch)
# and EIGEN_RUNTIME_NO_MALLOC (routing Eigen's runtime malloc gate into the
# counting hook) -- replicating the define set the retired benchmarks/ alloc-gate
# factory carried. Without ARGMIN_BENCH_TRACE_ALLOC the header compiles to its
# no-op twin and every gate passes vacuously; the alloc_sensor_liveness_canary
# case is the runtime proof the LIVE branch is selected. These are argmin-only
# gates that need no comparison library, so they run under the plain `dev` preset.
add_executable(argmin_alloc_gate_test
    alloc_gate_test.cpp
    alloc_sensor_counters.cpp
)
# Sensor B (the glibc __libc_* strong-symbol overrides) links only on Linux; the
# CMake platform guard is the sole gate (no __linux__ #ifdef inside the source).
# Elsewhere read_alloc_count() degrades to the always-live Eigen sensor.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(argmin_alloc_gate_test PRIVATE alloc_sensor_b_linux.cpp)
endif()
target_link_libraries(argmin_alloc_gate_test
    PRIVATE
        argmin::argmin
        Catch2::Catch2WithMain
)
target_compile_definitions(argmin_alloc_gate_test
    PRIVATE ARGMIN_BENCH_TRACE_ALLOC=1 EIGEN_RUNTIME_NO_MALLOC)
target_compile_options(argmin_alloc_gate_test
    PRIVATE ${ARGMIN_TEST_RELEASE_FLAGS} ${ARGMIN_WARNING_FLAGS})
# Sensor B overrides malloc/free with __libc_* forwards, which bypasses and
# fights the sanitizer's own allocator -- the binary segfaults at Catch2 test
# discovery under -fsanitize={address,memory}. And an allocation count taken
# under a sanitizer-replaced allocator is not a valid measurement regardless.
# So under a sanitizer build the gate is neither built nor discovered (the main
# argmin_unit_tests suite, including the [move_asan] cases, still runs under
# -fsanitize). Discovery is guarded alongside EXCLUDE_FROM_ALL: catch_discover
# on an unbuilt target otherwise registers _NOT_BUILT "Not Run" ctest failures.
# Both sanitize CI legs carry -fsanitize on CMAKE_CXX_FLAGS, the sole gate.
string(FIND "${CMAKE_CXX_FLAGS}" "-fsanitize" _argmin_alloc_gate_sanitizer)
if(_argmin_alloc_gate_sanitizer EQUAL -1)
    # The 14 gate cases carry the [alloc-gate] tag and the ctest `alloc-gate`
    # label; the liveness canary carries [canary] and is registered without the
    # label so the labeled count stays exactly the number of gates.
    catch_discover_tests(argmin_alloc_gate_test
        TEST_SPEC "[alloc-gate]"
        PROPERTIES LABELS "alloc-gate")
    catch_discover_tests(argmin_alloc_gate_test
        TEST_SPEC "[canary]")
else()
    set_target_properties(argmin_alloc_gate_test PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

# Dedicated ctest label for the move-and-step ASan instrument so the
# sanitize CI job can select it explicitly (`ctest -L asan-move`) even
# though the whole suite already runs under -fsanitize=address. This is
# a second discovery pass over the same executable restricted to the
# [move_asan]-tagged cases; it registers duplicate ctest entries for
# those cases (harmless -- they just run twice under a plain `ctest`).
# The augmented_lagrangian case is now move-safe and runs here as an
# ordinary passing case alongside the other policies.
catch_discover_tests(argmin_unit_tests
    TEST_SPEC "[move_asan]"
    TEST_PREFIX "asan-move:"
    PROPERTIES LABELS "asan-move"
)

# Dedicated ctest label for oracle-backed pins (wave-2 instruments:
# PRIMA / purecma / scipy CSV comparisons). No test carries the
# [oracle-pin] tag yet -- catch_discover_tests re-runs the executable's
# test list at build time, so tagging a TEST_CASE [oracle-pin] in a
# later plan is picked up automatically without another CMake edit.
catch_discover_tests(argmin_unit_tests
    TEST_SPEC "[oracle-pin]"
    TEST_PREFIX "oracle-pin:"
    PROPERTIES LABELS "oracle-pin"
)

# Dedicated ctest selection for the behavioral failure-mode suite. The
# [robustness]-tagged cases drive the reject / error / NaN-guard paths
# (cooperative abort, restoration terminals, filter-restoration legs,
# multi-start restart, the CCSA dual solve, and the QP augmentation
# recovery). This second discovery pass re-registers those cases under a
# "robustness:" name prefix so `ctest -R robustness` (and the label
# "robustness") select the suite; a plain ctest runs them once as usual.
catch_discover_tests(argmin_unit_tests
    TEST_SPEC "[robustness]"
    TEST_PREFIX "robustness:"
    PROPERTIES LABELS "robustness"
)

# Oracle CSV plumbing (wave-2 instruments, populated by a later plan).
# Configure-time copy: mirrors tests/oracles/ into this target's ctest
# working directory so oracle-pin cases can read checked-in CSVs by a
# relative path. No-ops until tests/oracles/ exists; re-run cmake after
# adding oracle files so the copy picks them up.
set(ARGMIN_ORACLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../oracles")
if(EXISTS "${ARGMIN_ORACLE_DIR}")
    file(COPY "${ARGMIN_ORACLE_DIR}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# The BOBYQA convergence oracle is loaded by name by the convergence /
# eval-count pin, so copy it explicitly (mirroring the cobyla_convergence.csv
# plumbing) and fail loudly at configure time if the generator step was
# skipped, rather than surfacing as a silent relative-path miss at run time.
set(ARGMIN_BOBYQA_ORACLE "${ARGMIN_ORACLE_DIR}/bobyqa_convergence.csv")
if(EXISTS "${ARGMIN_BOBYQA_ORACLE}")
    configure_file(
        "${ARGMIN_BOBYQA_ORACLE}"
        "${CMAKE_CURRENT_BINARY_DIR}/oracles/bobyqa_convergence.csv"
        COPYONLY)
endif()
