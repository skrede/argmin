# Full flashable NUCLEO-H753ZI image: CMSIS-only bare-metal (no HAL/RTOS),
# hand-written linker script, the _sbrk allocation instrument, and the
# four-policy fixed-N solve() probe with a semihosting report.
#
# Built only in the default firmware mode (MCU_LINK_GATE_ONLY=OFF). The
# on-device allocs/step number is an operator capture (see README.md); this
# CMake verifies the image cross-compiles and links to a valid .elf that fits
# the part.

enable_language(ASM)

# --- CMSIS-Core 6.x + STM32H7 device pack (headers + gcc startup + system) ---
# Mirrors the Eigen FetchContent convention: header/asm trees, no CMakeLists to
# suppress. Re-check the releases pages before bumping these tags. Only
# CMSIS/Core/Include is put on the include path, so no RTOS/DSP/NN code is
# pulled in.
#
# Fetch is the default, but an integrator porting this proof onto their own
# board almost always already has a CMSIS core + device pack from their SDK
# (STM32Cube, vendor HAL, Zephyr, PlatformIO). Set ARGMIN_MCU_CMSIS_CORE_DIR
# and/or ARGMIN_MCU_CMSIS_DEVICE_DIR to those source roots to bypass the clone
# and use the provisioned pack instead. Each is independent.
set(ARGMIN_MCU_CMSIS_CORE_DIR "" CACHE PATH
    "Pre-provisioned CMSIS-Core source root; when set, skips the cmsis_core FetchContent")
set(ARGMIN_MCU_CMSIS_DEVICE_DIR "" CACHE PATH
    "Pre-provisioned STM32H7 CMSIS device source root; when set, skips the cmsis_device_h7 FetchContent")

set(_cmsis_to_fetch "")
if(ARGMIN_MCU_CMSIS_CORE_DIR)
    set(cmsis_core_SOURCE_DIR "${ARGMIN_MCU_CMSIS_CORE_DIR}")
    message(STATUS "argmin_mcu: using provided CMSIS-Core at ${cmsis_core_SOURCE_DIR} (fetch skipped)")
else()
    FetchContent_Declare(cmsis_core
        GIT_REPOSITORY https://github.com/ARM-software/CMSIS_6.git
        GIT_TAG v6.3.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR this-directory-does-not-exist)
    list(APPEND _cmsis_to_fetch cmsis_core)
endif()
if(ARGMIN_MCU_CMSIS_DEVICE_DIR)
    set(cmsis_device_h7_SOURCE_DIR "${ARGMIN_MCU_CMSIS_DEVICE_DIR}")
    message(STATUS "argmin_mcu: using provided CMSIS device pack at ${cmsis_device_h7_SOURCE_DIR} (fetch skipped)")
else()
    FetchContent_Declare(cmsis_device_h7
        GIT_REPOSITORY https://github.com/STMicroelectronics/cmsis-device-h7.git
        GIT_TAG v1.10.7
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR this-directory-does-not-exist)
    list(APPEND _cmsis_to_fetch cmsis_device_h7)
endif()
if(_cmsis_to_fetch)
    FetchContent_MakeAvailable(${_cmsis_to_fetch})
endif()

set(CMSIS_CORE_INC "${cmsis_core_SOURCE_DIR}/CMSIS/Core/Include")
set(CMSIS_DEV_INC  "${cmsis_device_h7_SOURCE_DIR}/Include")
set(CMSIS_STARTUP  "${cmsis_device_h7_SOURCE_DIR}/Source/Templates/gcc/startup_stm32h753xx.s")
set(CMSIS_SYSTEM   "${cmsis_device_h7_SOURCE_DIR}/Source/Templates/system_stm32h7xx.c")

# Eigen stack-temporary ceiling: dynamic temporaries above this fall back from
# alloca (stack) to the heap at runtime (where the alloc gate reports them as
# an honest red -- detectable, not a crash). The ceiling only exists because
# ARGMIN_EIGEN_MCU_DEFS defines EIGEN_ALLOCA explicitly: at strict -std=c++20
# (CMAKE_CXX_EXTENSIONS OFF) newlib hides its alloca macro and Eigen's
# auto-detection finds nothing, making every internal kernel temporary a heap
# malloc and this limit dead code. That misconfiguration was the measured
# kraft_slsqp 2.80 allocs/step on the NUCLEO (two triangular-solve blocking
# buffers per inner QP iteration, 79+223 B at N=4; 0.00 on host because
# __linux__ short-circuits the check) and why sweeping the limit left the
# binary byte-identical -- see nucleo_h753zi/onchip-result.md.
#
# Budget: the limit gates two object classes, both measured (sweep-result.md,
# 2026-07-18, arm-gcc 15.2.Rel1). (1) Runtime dynamic temporaries via the
# alloca-vs-heap ternary in ei_declare_aligned_stack_constructed_variable --
# largest is 223 B (kraft's lsi.h triangular solve; nw_sqp/filter_nw_sqp/lm
# request none at these fixed N). (2) Fixed-size Eigen stack objects via a
# compile-time static_assert(Size*sizeof(T) <= LIMIT) in DenseStorage.h --
# largest is 512 B (the 8x8 TriangularMatrixMatrix panel buffer), which sets a
# hard COMPILE FLOOR: any limit < 512 fails to build (verified: 256 -> static
# assert (512 <= 256)). 8192 clears both -- 16x the 512 B compile floor and
# ~36.7x the 223 B dynamic max -- so the margin swamps codegen/toolchain deltas.
# The live sweep (512/1024/8192 produce distinct binaries; 256 fails to compile)
# proves the knob is not dead code. Do not raise it toward the reserve size, and
# do not re-tune it tight. Rule + full sweep: docs/embedded.md, sweep-result.md.
set(MCU_EIGEN_STACK_LIMIT 8192 CACHE STRING
    "EIGEN_STACK_ALLOCATION_LIMIT for the fixed-N MCU probe (bytes)")

add_executable(nucleo_h753zi_probe
    nucleo_main.cpp
    wrap_malloc.cpp
    usart3_console.cpp
    "${CMSIS_SYSTEM}"
    "${CMSIS_STARTUP}")

target_include_directories(nucleo_h753zi_probe SYSTEM PRIVATE
    "${ARGMIN_INCLUDE_DIR}"
    "${eigen3_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}"        # mcu/ root, for "probe/rt_probe_workload.h"
    "${CMSIS_CORE_INC}"
    "${CMSIS_DEV_INC}")

target_compile_definitions(nucleo_h753zi_probe PRIVATE
    STM32H753xx
    ARGMIN_BENCH_TRACE_ALLOC=1
    ARGMIN_ALLOC_GATE_EXPECT_ZERO
    ${ARGMIN_EIGEN_MCU_DEFS}
    EIGEN_STACK_ALLOCATION_LIMIT=${MCU_EIGEN_STACK_LIMIT})

target_compile_options(nucleo_h753zi_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -fno-exceptions -fno-rtti -Wno-psabi>
    -ffunction-sections -fdata-sections
    -fstack-usage)               # emits .su files for the stack-limit sweep

target_link_options(nucleo_h753zi_probe PRIVATE
    ${ARGMIN_M7_LINK_ARCH}
    -T "${CMAKE_CURRENT_SOURCE_DIR}/stm32h753zi_flash.ld"
    -nostartfiles
    --specs=nano.specs --specs=nosys.specs
    # newlib-nano trims printf: pull in the float-capable variant so the report
    # renders %zu / %.2f (allocs/step) instead of literal "zu".
    -u _printf_float
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
