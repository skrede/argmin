set(ARGMIN_PYTHON_ENV "${CMAKE_SOURCE_DIR}/.venv" CACHE PATH
    "Repository-local interpreter environment the extension is built and tested against")

if(WIN32)
    set(_argmin_python_interpreter "${ARGMIN_PYTHON_ENV}/Scripts/python.exe")
else()
    set(_argmin_python_interpreter "${ARGMIN_PYTHON_ENV}/bin/python")
endif()

# Pinning the interpreter before find_package is deliberate: a build that fell
# back to whatever interpreter is on PATH would link the extension against one
# runtime and run the suite under another, and the suite would report on a
# module nobody built.
if(NOT EXISTS "${_argmin_python_interpreter}")
    message(FATAL_ERROR
        "No interpreter at ${_argmin_python_interpreter}. Create it with:\n"
        "    python3 scripts/bootstrap_python_env.py\n"
        "or point ARGMIN_PYTHON_ENV at an existing environment.")
endif()

set(Python_EXECUTABLE "${_argmin_python_interpreter}")
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

if(ARGMIN_CMAKE_FETCH_DEPS)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.12.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
        SYSTEM
    )
    FetchContent_MakeAvailable(nanobind)
else()
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _argmin_nanobind_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _argmin_nanobind_locate_result
    )
    if(NOT _argmin_nanobind_locate_result EQUAL 0)
        message(FATAL_ERROR
            "nanobind is not installed in ${Python_EXECUTABLE}. Install it there, "
            "or configure with -DARGMIN_CMAKE_FETCH_DEPS=ON.")
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${_argmin_nanobind_cmake_dir}")
    find_package(nanobind CONFIG REQUIRED)
endif()

# CPython's and nanobind's headers are macro-heavy: under the project warning set
# their expansions report hundreds of C-style casts and signedness conversions
# from code nobody here wrote. Consuming them as system headers keeps the whole
# warning set live on the binding sources themselves -- the same treatment lib/
# already gives Eigen.
function(argmin_python_consume_as_system target)
    if(NOT TARGET ${target})
        return()
    endif()
    get_target_property(_includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_includes)
        set_target_properties(${target} PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_includes}")
    endif()
endfunction()

argmin_python_consume_as_system(Python::Module)

nanobind_add_module(
    _argmin
    NB_STATIC
    src/_argmin.cpp
    src/bindings/vocabulary_bindings.cpp
    src/bindings/qp_bindings.cpp
    src/bindings/nlp_gradient_bindings.cpp
)

argmin_python_consume_as_system(nanobind-static)

target_include_directories(_argmin PRIVATE src)

target_link_libraries(_argmin PRIVATE argmin::argmin)

target_compile_features(_argmin PRIVATE cxx_std_20)

target_compile_definitions(_argmin PRIVATE
    ARGMIN_PYTHON_VERSION_STRING="${PROJECT_VERSION}")

target_compile_options(_argmin PRIVATE ${ARGMIN_WARNING_FLAGS})

set_target_properties(_argmin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/argmin")

install(TARGETS _argmin LIBRARY DESTINATION argmin)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/argmin/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/argmin/py.typed"
    DESTINATION argmin)

add_test(
    NAME python_bindings_pytest
    COMMAND "${Python_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests")

set_tests_properties(python_bindings_pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR};PYTHONNOUSERSITE=1")
