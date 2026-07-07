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
    scheduling_test.cpp
    convergence_test.cpp
    line_search_test.cpp
    multistart_test.cpp
    projected_gn_test.cpp
    solver_group_test.cpp
    kraft_slsqp_test.cpp
    kraft_lsq_qp_test.cpp
    kraft_lsq_qp_recovery_test.cpp
    basic_solver_test.cpp
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
    isres_best_track_test.cpp
    halton_distinctness_test.cpp
    mma_coefficient_pin_test.cpp
    options_test.cpp
    slsqp_line_search_witness_test.cpp
    nw_sqp_soc_witness_test.cpp
    kraft_sigma_bump_witness_test.cpp
    merit_function_test.cpp
    substrate_scale_witness_test.cpp
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
