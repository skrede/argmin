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

if(PROJECT_IS_TOP_LEVEL AND NOT ARGMIN_CMAKE_FETCH_DEPS)
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

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/argminConfigVersion.cmake"
        DESTINATION ${argmin_INSTALL_CMAKEDIR}
    )
endif()
