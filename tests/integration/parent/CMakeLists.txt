cmake_minimum_required(VERSION 3.28)
project(argmin_transitive_parent VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Must be set before argmin's directory is added: its option() yields to an existing normal variable,
# and its install rules are decided while that directory is processed.
set(ARGMIN_INSTALL ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../.. ${CMAKE_CURRENT_BINARY_DIR}/argmin)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(transitive_parent STATIC probe.cpp)
add_library(transitive_parent::transitive_parent ALIAS transitive_parent)

target_include_directories(transitive_parent
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(transitive_parent
    PUBLIC
    argmin::argmin
)

install(TARGETS transitive_parent
    EXPORT transitive_parentTargets
)

install(DIRECTORY include/transitive_parent/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/transitive_parent
)

set(transitive_parent_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/transitive_parent)

install(EXPORT transitive_parentTargets
    FILE transitive_parentTargets.cmake
    NAMESPACE transitive_parent::
    DESTINATION ${transitive_parent_INSTALL_CMAKEDIR}
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/transitive_parentConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/transitive_parentConfig.cmake"
    INSTALL_DESTINATION ${transitive_parent_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/transitive_parentConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/transitive_parentConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/transitive_parentConfigVersion.cmake"
    DESTINATION ${transitive_parent_INSTALL_CMAKEDIR}
)
