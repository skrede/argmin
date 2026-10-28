set(argmin_fuzz_corpus_dir ${CMAKE_CURRENT_BINARY_DIR}/corpus)

# Pinned on the targets rather than left to the build type. Without -fno-sanitize-recover an undefined-behavior
# report lets execution continue and the run exits zero; without -g a finding names no source line; at the -O0 of a
# Debug configure the throughput drops by more than an order of magnitude.
set(argmin_fuzz_flags
    -fsanitize=fuzzer,address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer
    -O1
    -g
)

function(argmin_add_fuzz_binary target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
    target_link_libraries(${target} PRIVATE argmin::argmin)
    target_compile_options(${target} PRIVATE ${argmin_fuzz_flags} ${ARGMIN_WARNING_FLAGS})
    target_link_options(${target} PRIVATE ${argmin_fuzz_flags})
endfunction()

# libFuzzer writes every coverage-increasing input into the directory it is given, so runs only ever see this
# build-tree copy and never the committed seeds. The copy is additive.
function(argmin_add_fuzz_target target minimum)
    argmin_add_fuzz_binary(${target} ${target}.cpp)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/corpus/${target} DESTINATION ${argmin_fuzz_corpus_dir})
    add_test(NAME ${target}_replay
             COMMAND ${CMAKE_COMMAND}
                     -DTARGET=$<TARGET_FILE:${target}>
                     -DCORPUS=${argmin_fuzz_corpus_dir}/${target}
                     -DMINIMUM=${minimum}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/replay.cmake)
    set_tests_properties(${target}_replay PROPERTIES ENVIRONMENT "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1")
endfunction()

argmin_add_fuzz_target(fuzz_dense_admm_qp 15)
argmin_add_fuzz_target(fuzz_dense_active_set_qp 14)
argmin_add_fuzz_target(fuzz_sparse_admm_qp 11)

# The same target with one oracle leg inverted, kept out of the replay tests; CI requires it to fail.
argmin_add_fuzz_binary(fuzz_dense_admm_qp_planted fuzz_dense_admm_qp.cpp)
target_compile_definitions(fuzz_dense_admm_qp_planted PRIVATE ARGMIN_FUZZ_PLANT_VIOLATION)
