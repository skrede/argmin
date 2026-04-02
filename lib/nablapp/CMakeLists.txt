if(NABLAPP_CMAKE_FETCH_DEPS)
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
        message(FATAL_ERROR "nablapp requires Eigen >= 3.4, found ${Eigen3_VERSION}")
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

add_library(nablapp INTERFACE)
add_library(nablapp::nablapp ALIAS nablapp)

target_include_directories(nablapp
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIRECTORY}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(nablapp
    INTERFACE
    Eigen3::Eigen
)

if(PROJECT_IS_TOP_LEVEL AND NOT NABLAPP_CMAKE_FETCH_DEPS)
    install(TARGETS nablapp
        EXPORT nablappTargets
    )

    install(DIRECTORY ${INCLUDE_DIRECTORY}/nablapp/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nablapp
    )

    set(nablapp_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/nablapp)

    install(EXPORT nablappTargets
        FILE nablappTargets.cmake
        NAMESPACE nablapp::
        DESTINATION ${nablapp_INSTALL_CMAKEDIR}
    )

    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/nablappConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/nablappConfig.cmake"
        INSTALL_DESTINATION ${nablapp_INSTALL_CMAKEDIR}
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/nablappConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nablappConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nablappConfigVersion.cmake"
        DESTINATION ${nablapp_INSTALL_CMAKEDIR}
    )
endif()
