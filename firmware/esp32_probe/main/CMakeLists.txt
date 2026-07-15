idf_component_register(SRCS "probe_main.cpp"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_driver_uart heap)

# argmin headers + Eigen + the firmware/ root (for probe/rt_probe_workload.h),
# bridged from the top-level project CMakeLists via CACHE variables.
target_include_directories(${COMPONENT_LIB} SYSTEM PRIVATE
    "${ARGMIN_PROBE_INCLUDE_DIR}"
    "${ARGMIN_PROBE_EIGEN_SOURCE_DIR}"
    "${ARGMIN_PROBE_FIRMWARE_DIR}")

# ESP-IDF v6 defaults C++ to -std=gnu++26; GCC honors the LAST -std=, so this
# per-component override (appended after IDF's global flag) wins. PRIVATE so it
# never leaks into IDF's generated C TUs (61-RESEARCH-esp32.md §1).
target_compile_options(${COMPONENT_LIB} PRIVATE
    -std=gnu++20 -Wno-psabi)

# Eigen bare-metal-ish macro set. EIGEN_RUNTIME_NO_MALLOC + the counting
# eigen_assert are defined in probe_main.cpp itself (before the first Eigen
# include), so the portable Eigen-native allocation sensor works exactly as on
# host; do NOT define EIGEN_NO_MALLOC here (it would hard-abort instead).
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    ARGMIN_BENCH_TRACE_ALLOC=1
    ARGMIN_ALLOC_GATE_EXPECT_ZERO
    EIGEN_DONT_VECTORIZE
    EIGEN_MAX_ALIGN_BYTES=8)
