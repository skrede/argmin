# Full flashable NUCLEO-H753ZI image: CMSIS-only bare-metal (no HAL/RTOS),
# hand-written linker script, the _sbrk allocation instrument, and the
# four-policy fixed-N solve() probe with a semihosting report (MCU-01/03/04).
#
# Built only in the default firmware mode (MCU_LINK_GATE_ONLY=OFF). The
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

# Eigen stack-temporary ceiling: dynamic temporaries above this fall back from
# alloca (stack) to the heap at runtime. Set to 64 KiB (host-like, and well
# within the DTCM stack budget below) rather than an over-tight value. NOTE:
# raising this does NOT explain kraft_slsqp's on-device per-step allocation --
# sweeping it 8192 -> 65536 left the binary byte-identical and kraft still
# heap-allocated 2.80x/step on the NUCLEO. kraft's residual is a genuine
# arm-none-eabi/newlib-specific allocation in its Kraft-LSEI QP substrate
# (Phase 55 drove it to 0.00/step on host only; not reproducible on host even
# with these exact flags), tracked as a follow-up finding -- see
# nucleo_h753zi/onchip-result.md. The other three RT policies are 0.00/step.
set(MCU_EIGEN_STACK_LIMIT 65536 CACHE STRING
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
