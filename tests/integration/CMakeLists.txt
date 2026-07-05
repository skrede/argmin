# Exercises the packaged-consumer path: stage-installs the parent project into a
# local prefix, then configures and builds the consumer/ subdirectory against that
# staged install via find_package(argmin CONFIG REQUIRED) -- the exact mechanism a
# downstream/distro consumer uses. Only entered when ARGMIN_CMAKE_FETCH_DEPS is OFF
# (system Eigen path); see tests/CMakeLists.txt.

include(ExternalProject)

set(ARGMIN_INTEGRATION_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/stage")

# Stage-install the parent project's headers and CMake package config so the
# consumer subproject below can find_package(argmin CONFIG REQUIRED) exactly as
# a downstream user would, without a separate manual install step.
add_custom_target(argmin_integration_stage
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --prefix "${ARGMIN_INTEGRATION_STAGE_DIR}"
    COMMENT "Staging argmin install tree for the integration consumer test"
    VERBATIM
)

ExternalProject_Add(argmin_integration_consumer
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/consumer"
    CMAKE_ARGS
        "-DCMAKE_PREFIX_PATH=${ARGMIN_INTEGRATION_STAGE_DIR}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE
    DEPENDS argmin_integration_stage
)

ExternalProject_Get_Property(argmin_integration_consumer BINARY_DIR)

add_test(
    NAME integration_consumer_test
    COMMAND "${BINARY_DIR}/integration_test"
)
