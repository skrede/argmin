# Full flashable NUCLEO-H753ZI image: CMSIS-only bare-metal (no HAL/RTOS),
# hand-written linker script, the _sbrk allocation instrument, and the
# four-policy fixed-N solve() probe with a semihosting report (MCU-01/03/04).
#
# Built only in the default firmware mode (FIRMWARE_LINK_GATE_ONLY=OFF). The
# on-device allocs/step number is an operator capture (see README.md); this
# CMake verifies the image cross-compiles and links to a valid .elf that fits
# the part.

enable_language(ASM)

# --- CMSIS-Core 6.x + STM32H7 device pack (headers + gcc startup + system) ---
# Mirrors the Eigen FetchContent convention: header/asm trees, no CMakeLists to
# suppress. Re-check the releases pages before bumping these tags
# (61-RESEARCH-nucleo.md §1). Only CMSIS/Core/Include is put on the include
# path, so no RTOS/DSP/NN code is pulled in.
FetchContent_Declare(cmsis_core
    GIT_REPOSITORY https://github.com/ARM-software/CMSIS_6.git
    GIT_TAG v6.3.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
    SOURCE_SUBDIR this-directory-does-not-exist)
FetchContent_Declare(cmsis_device_h7
    GIT_REPOSITORY https://github.com/STMicroelectronics/cmsis-device-h7.git
    GIT_TAG v1.10.7
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
    SOURCE_SUBDIR this-directory-does-not-exist)
FetchContent_MakeAvailable(cmsis_core cmsis_device_h7)

set(CMSIS_CORE_INC "${cmsis_core_SOURCE_DIR}/CMSIS/Core/Include")
set(CMSIS_DEV_INC  "${cmsis_device_h7_SOURCE_DIR}/Include")
set(CMSIS_STARTUP  "${cmsis_device_h7_SOURCE_DIR}/Source/Templates/gcc/startup_stm32h753xx.s")
set(CMSIS_SYSTEM   "${cmsis_device_h7_SOURCE_DIR}/Source/Templates/system_stm32h7xx.c")

# Eigen fixed-N stack-temporary ceiling. Default (128 KiB) is the entire DTCM
# budget; swept against the real build (61-02 task 6): the largest single Eigen
# fixed-size stack temporary at these fixed N is <= 4096 bytes (both 8192 and
# 4096 compile clean), so 8192 is a 2x-margin pin that still catches a gross
# large-temporary regression while sitting 16x below Eigen's default. (Note the
# measured worst-case total call frame is ~19888 bytes -- that is a function
# frame, not a single Eigen temporary, so it does not gate this limit; the
# linker stack reserve below covers it.) Overridable to re-pin without edits.
set(FIRMWARE_EIGEN_STACK_LIMIT 8192 CACHE STRING
    "EIGEN_STACK_ALLOCATION_LIMIT for the fixed-N MCU probe (bytes)")

add_executable(nucleo_h753zi_probe
    nucleo_main.cpp
    wrap_malloc.cpp
    "${CMSIS_SYSTEM}"
    "${CMSIS_STARTUP}")

target_include_directories(nucleo_h753zi_probe SYSTEM PRIVATE
    "${ARGMIN_INCLUDE_DIR}"
    "${eigen3_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}"        # firmware/ root, for "probe/rt_probe_workload.h"
    "${CMSIS_CORE_INC}"
    "${CMSIS_DEV_INC}")

target_compile_definitions(nucleo_h753zi_probe PRIVATE
    STM32H753xx
    ARGMIN_BENCH_TRACE_ALLOC=1
    ARGMIN_ALLOC_GATE_EXPECT_ZERO
    ${ARGMIN_EIGEN_MCU_DEFS}
    EIGEN_STACK_ALLOCATION_LIMIT=${FIRMWARE_EIGEN_STACK_LIMIT})

target_compile_options(nucleo_h753zi_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -fno-exceptions -fno-rtti -Wno-psabi>
    -ffunction-sections -fdata-sections
    -fstack-usage)               # emits .su files for the stack-limit sweep

target_link_options(nucleo_h753zi_probe PRIVATE
    ${ARGMIN_M7_LINK_ARCH}
    -T "${CMAKE_CURRENT_SOURCE_DIR}/stm32h753zi_flash.ld"
    -nostartfiles
    --specs=nano.specs --specs=rdimon.specs
    # Intercept the malloc family so the on-target sensor counts every
    # allocation event (wrap_malloc.cpp), mirroring the host strong-symbol gate.
    -Wl,--wrap=malloc,--wrap=free,--wrap=calloc,--wrap=realloc
    -Wl,--gc-sections
    -Wl,-Map=nucleo_h753zi_probe.map)

set_target_properties(nucleo_h753zi_probe PROPERTIES SUFFIX ".elf")

# Report size at build time (must fit 2 MB flash / 128 KB DTCM).
find_program(ARGMIN_ARM_SIZE "${ARGMIN_ARM_TOOLCHAIN_PREFIX}size")
if(ARGMIN_ARM_SIZE)
    add_custom_command(TARGET nucleo_h753zi_probe POST_BUILD
        COMMAND "${ARGMIN_ARM_SIZE}" "$<TARGET_FILE:nucleo_h753zi_probe>"
        VERBATIM)
endif()
