# Standalone firmware/ sibling project for the on-device embeddability proof.
#
# This is its OWN project() below the include boundary: it consumes the argmin
# headers via an include path only and is NEVER added to the header-only
# library's include path (MCU-01). It is not built by the top-level argmin
# CMake; it is configured separately with the arm-none-eabi toolchain file:
#
#   # minimal -fno-exceptions cross-compile-link CI gate (no board, no BSP):
#   cmake -S firmware -B build-arm-gate \
#     -DCMAKE_TOOLCHAIN_FILE=firmware/cmake/toolchain-arm-none-eabi.cmake \
#     -DFIRMWARE_LINK_GATE_ONLY=ON
#   cmake --build build-arm-gate
#
#   # full flashable NUCLEO-H753ZI image (CMSIS + linker script + instrument):
#   cmake -S firmware -B build-arm-nucleo \
#     -DCMAKE_TOOLCHAIN_FILE=firmware/cmake/toolchain-arm-none-eabi.cmake
#   cmake --build build-arm-nucleo

cmake_minimum_required(VERSION 3.28)
project(argmin_firmware CXX)

# When ON, define only the board-free -fno-exceptions link gate (the per-commit
# CI target): no CMSIS FetchContent, no linker script, no startup. This keeps
# board-specific dependencies and their flake out of the per-commit gate.
option(FIRMWARE_LINK_GATE_ONLY
    "Build only the board-free -fno-exceptions arm cross-compile-link gate" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# argmin headers, consumed header-only. The firmware includes the library; the
# library never includes the firmware.
set(ARGMIN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/argmin/include")

# Eigen, pinned to the same tag the library uses (lib/argmin/CMakeLists.txt).
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
    SOURCE_SUBDIR this-directory-does-not-exist
)
FetchContent_MakeAvailable(Eigen3)

# Cortex-M7 hard-float arch flags. The toolchain file already applies these at
# compile time (CMAKE_CXX_FLAGS_INIT); the gcc driver also needs them on the
# link line to select the correct newlib-nano multilib.
set(ARGMIN_M7_LINK_ARCH
    -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb)

# Eigen bare-metal macros common to every firmware TU (research 61-RESEARCH-nucleo §6).
# Cortex-M7 has no NEON and FPv5 is scalar-only: no vector ISA to align for, so
# EIGEN_MAX_ALIGN_BYTES=8 (natural double) removes the aligned-allocation path.
set(ARGMIN_EIGEN_MCU_DEFS
    EIGEN_DONT_VECTORIZE
    EIGEN_MAX_ALIGN_BYTES=8)

# --- MCU-02: board-free -fno-exceptions -fno-rtti cross-compile-link gate ----
#
# Reuses the shared four-policy RT workload (probe/rt_probe_workload.h). Built
# WITHOUT ARGMIN_BENCH_TRACE_ALLOC, so it is a pure instantiation + execution
# surface. nano.specs+nosys.specs supply the C library and weak syscall stubs
# (_sbrk/_write) so the link needs no linker script, startup, or CMSIS.
add_executable(argmin_arm_link_gate link_gate_main.cpp)
target_include_directories(argmin_arm_link_gate SYSTEM PRIVATE
    "${ARGMIN_INCLUDE_DIR}"
    "${eigen3_SOURCE_DIR}")
target_compile_options(argmin_arm_link_gate PRIVATE
    -O2 -fno-exceptions -fno-rtti -Wno-psabi)
target_compile_definitions(argmin_arm_link_gate PRIVATE ${ARGMIN_EIGEN_MCU_DEFS})
target_link_options(argmin_arm_link_gate PRIVATE
    ${ARGMIN_M7_LINK_ARCH}
    --specs=nano.specs --specs=nosys.specs
    -Wl,-Map=argmin_arm_link_gate.map)
set_target_properties(argmin_arm_link_gate PROPERTIES SUFFIX ".elf")

# --- MCU-01/03/04: full flashable NUCLEO-H753ZI image ------------------------
if(NOT FIRMWARE_LINK_GATE_ONLY
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nucleo_h753zi/CMakeLists.txt")
    add_subdirectory(nucleo_h753zi)
endif()
