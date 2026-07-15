# Top-level ESP-IDF project for the ESP32 *accessible* allocation proof.
#
# This is the D-01 accessible claim: the fixed-N four-policy solve() probe runs
# under ESP-IDF with FreeRTOS present and IDF's allocator in the loop -- a real
# embeddability result but a SOFTER zero-heap statement than the bare-metal
# NUCLEO proof (mcu/nucleo_h753zi/). It is never presented as bare-metal.
#
# Build (operator or CI-with-IDF):
#   . /opt/esp-idf/export.sh
#   idf.py -C mcu/esp32_probe set-target esp32
#   idf.py -C mcu/esp32_probe build       # -> build/argmin_esp32_probe.bin
#   idf.py -C mcu/esp32_probe flash monitor
#
# Eigen FetchContent MUST run here, BEFORE the IDF project.cmake include:
# ESP-IDF pre-scans every component's CMakeLists in cmake -P script mode to
# harvest REQUIRES, and FetchContent_Declare calls define_property(), which is
# not scriptable -- doing the fetch inside main/CMakeLists.txt fails configure
# (verified in 61-RESEARCH-esp32.md §1). The source dir is bridged into the
# component via a CACHE variable.

cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR this-directory-does-not-exist
)
FetchContent_MakeAvailable(Eigen3)
set(ARGMIN_PROBE_EIGEN_SOURCE_DIR "${eigen3_SOURCE_DIR}" CACHE PATH
    "Eigen source dir, bridged into the main component")

# argmin headers, consumed header-only (the firmware includes the library; the
# library never includes the firmware -- MCU-01 boundary).
set(ARGMIN_PROBE_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../lib/argmin/include"
    CACHE PATH "argmin header include dir")
set(ARGMIN_PROBE_MCU_DIR "${CMAKE_CURRENT_LIST_DIR}/.."
    CACHE PATH "mcu/ root, for probe/rt_probe_workload.h")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
idf_build_set_property(MINIMAL_BUILD ON)
project(argmin_esp32_probe)
