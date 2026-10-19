if(ARGMIN_CMAKE_FETCH_DEPS)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR this-directory-does-not-exist
    )
    FetchContent_MakeAvailable(Eigen3)
    add_library(eigen_headers INTERFACE)
    add_library(Eigen3::Eigen ALIAS eigen_headers)
    target_include_directories(eigen_headers SYSTEM INTERFACE
        "${eigen3_SOURCE_DIR}"
    )
    set(ARGMIN_EIGEN_INCLUDE_DIRS "${eigen3_SOURCE_DIR}" CACHE INTERNAL
        "Eigen include dirs, for tooling that needs them outside a target link")
else()
    find_package(Eigen3 CONFIG REQUIRED)
    if(Eigen3_VERSION VERSION_LESS "3.4")
        message(FATAL_ERROR "argmin requires Eigen >= 3.4, found ${Eigen3_VERSION}")
    endif()

    get_target_property(_eigen_aliased Eigen3::Eigen ALIASED_TARGET)
    if(_eigen_aliased)
        set(_eigen_target ${_eigen_aliased})
    else()
        set(_eigen_target Eigen3::Eigen)
    endif()
    get_target_property(_eigen_inc ${_eigen_target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${_eigen_target} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_eigen_inc}"
    )
    set(ARGMIN_EIGEN_INCLUDE_DIRS "${_eigen_inc}" CACHE INTERNAL
        "Eigen include dirs, for tooling that needs them outside a target link")
endif()

set(INCLUDE_DIRECTORY include)

add_library(argmin INTERFACE)
add_library(argmin::argmin ALIAS argmin)

target_include_directories(argmin
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIRECTORY}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(argmin
    INTERFACE
    Eigen3::Eigen
)

if(ARGMIN_INSTALL)
    if(eigen3_SOURCE_DIR)
        message(FATAL_ERROR
            "ARGMIN_INSTALL is ON but Eigen was fetched (ARGMIN_CMAKE_FETCH_DEPS). A fetched Eigen is a "
            "local target belonging to no export set, so argmin's export cannot generate around it. "
            "Supply a discoverable Eigen and leave ARGMIN_CMAKE_FETCH_DEPS off, or turn ARGMIN_INSTALL off.")
    endif()

    install(TARGETS argmin
        EXPORT argminTargets
    )

    install(DIRECTORY ${INCLUDE_DIRECTORY}/argmin/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/argmin
    )

    set(argmin_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/argmin)

    install(EXPORT argminTargets
        FILE argminTargets.cmake
        NAMESPACE argmin::
        DESTINATION ${argmin_INSTALL_CMAKEDIR}
    )

    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/argminConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfig.cmake"
        INSTALL_DESTINATION ${argmin_INSTALL_CMAKEDIR}
    )

    # At 0.x a same-major rule would accept this install for a request of 0.1.0, contradicting the
    # pre-1.0 policy that every minor release may break; staying stricter than semver past 1.0 is accepted.
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfigVersion.cmake"
        COMPATIBILITY SameMinorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfigVersion.cmake"
        DESTINATION ${argmin_INSTALL_CMAKEDIR}
    )
endif()
