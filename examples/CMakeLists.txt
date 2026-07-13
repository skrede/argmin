# Runnable examples, one target per problem class. Enable with
# -DARGMIN_BUILD_EXAMPLES=ON (dependencies auto-fetched when
# -DARGMIN_CMAKE_FETCH_DEPS=ON).

add_executable(unconstrained_lbfgsb unconstrained_lbfgsb.cpp)
target_link_libraries(unconstrained_lbfgsb PRIVATE argmin::argmin)

add_executable(constrained_slsqp constrained_slsqp.cpp)
target_link_libraries(constrained_slsqp PRIVATE argmin::argmin)

add_executable(least_squares_lm least_squares_lm.cpp)
target_link_libraries(least_squares_lm PRIVATE argmin::argmin)

add_executable(derivative_free_bobyqa derivative_free_bobyqa.cpp)
target_link_libraries(derivative_free_bobyqa PRIVATE argmin::argmin)

add_executable(step_level_control step_level_control.cpp)
target_link_libraries(step_level_control PRIVATE argmin::argmin)
