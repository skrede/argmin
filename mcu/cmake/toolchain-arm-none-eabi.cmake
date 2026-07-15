# CMake toolchain file for bare-metal Cortex-M7 (NUCLEO-H753ZI) cross-compile.
#
# Used by the standalone mcu/ project() for both the minimal
# -fno-exceptions link gate (MCU_LINK_GATE_ONLY=ON, no BSP) and the full
# NUCLEO image. It never touches the host build of the library or its tests.
#
# The compilers are not FetchContent-able; CI pins a documented arm-none-eabi
# GCC (14.x) installed via the runner package manager. The authoring host
# measured 16.1.0 -- the flag set below is version-stable across both.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Do not let CMake attempt a hosted executable link during its compiler probe:
# there is no startup/BSP at configure time, so a full link would fail. A
# static-library probe compiles without linking.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(ARGMIN_ARM_TOOLCHAIN_PREFIX "arm-none-eabi-" CACHE STRING
    "arm-none-eabi cross toolchain prefix")

find_program(ARGMIN_ARM_CC  "${ARGMIN_ARM_TOOLCHAIN_PREFIX}gcc" REQUIRED)
find_program(ARGMIN_ARM_CXX "${ARGMIN_ARM_TOOLCHAIN_PREFIX}g++" REQUIRED)

set(CMAKE_C_COMPILER   "${ARGMIN_ARM_CC}")
set(CMAKE_CXX_COMPILER "${ARGMIN_ARM_CXX}")
set(CMAKE_ASM_COMPILER "${ARGMIN_ARM_CC}")

# Cortex-M7 hard-float codegen. fpv5-d16 is the double+single-precision FPU the
# H753 actually has; fpv5-sp-d16 (single-only) would be wrong for double solvers.
set(ARGMIN_ARM_ARCH_FLAGS
    "-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb")

set(CMAKE_C_FLAGS_INIT   "${ARGMIN_ARM_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${ARGMIN_ARM_ARCH_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${ARGMIN_ARM_ARCH_FLAGS}")

# Search host system paths for programs only; libraries/headers come from the
# cross toolchain sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
