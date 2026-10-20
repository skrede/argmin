# The target decides, not the inheritable eigen3_SOURCE_DIR, because a parent may already supply
# Eigen3::Eigen. Fetching is opt-in, so without the option nothing is downloaded. The 3.4 floor is
# checked by hand: Eigen's same-major version file rejects a 3.4 request from Eigen 5.
if(NOT TARGET Eigen3::Eigen)
    if(ARGMIN_CMAKE_FETCH_DEPS)
        # A parent's ALWAYS would let find_package satisfy the declaration and leave eigen3_SOURCE_DIR empty.
        set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c  # 3.4.0
            EXCLUDE_FROM_ALL
            SOURCE_SUBDIR this-directory-does-not-exist
        )
        FetchContent_MakeAvailable(Eigen3)
        add_library(eigen_headers INTERFACE)
        add_library(Eigen3::Eigen ALIAS eigen_headers)
        target_include_directories(eigen_headers SYSTEM INTERFACE
            "${eigen3_SOURCE_DIR}"
        )
    else()
        find_package(Eigen3 CONFIG GLOBAL)
        if(NOT Eigen3_FOUND OR Eigen3_VERSION VERSION_LESS "3.4")
            message(FATAL_ERROR
                "argmin requires Eigen >= 3.4, and no discoverable Eigen3 package of that version was found. "
                "Install Eigen >= 3.4 where find_package can see it, or set ARGMIN_CMAKE_FETCH_DEPS=ON to "
                "build against the pinned source.")
        endif()
    endif()
endif()

get_target_property(_eigen_aliased Eigen3::Eigen ALIASED_TARGET)
if(_eigen_aliased)
    set(_eigen_target ${_eigen_aliased})
else()
    set(_eigen_target Eigen3::Eigen)
endif()
get_target_property(_eigen_inc ${_eigen_target} INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_eigen_imported ${_eigen_target} IMPORTED)
set(ARGMIN_EIGEN_INCLUDE_DIRS "${_eigen_inc}" CACHE INTERNAL
    "Eigen include dirs, for tooling that needs them outside a target link")

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
    if(NOT _eigen_imported)
        message(FATAL_ERROR
            "ARGMIN_INSTALL is ON but the Eigen3::Eigen that argmin links is not an imported target: it was "
            "fetched (ARGMIN_CMAKE_FETCH_DEPS) or built by a parent project. Such a target belongs to no "
            "export set, so argmin's export cannot generate around it. Supply a discoverable Eigen and "
            "leave ARGMIN_CMAKE_FETCH_DEPS off, or turn ARGMIN_INSTALL off.")
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
        ARCH_INDEPENDENT
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfigVersion.cmake"
        DESTINATION ${argmin_INSTALL_CMAKEDIR}
    )
endif()
