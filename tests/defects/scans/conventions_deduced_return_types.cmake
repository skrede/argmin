set(SCAN_NAME "the driver facades spell out the return type of every member function")
set(SCAN_FILES "lib/argmin/include/argmin/solver/stepper.h;lib/argmin/include/argmin/solver/step_budget_solver.h;lib/argmin/include/argmin/solver/time_budget_solver.h;lib/argmin/include/argmin/solver/step_and_time_budget_solver.h")
set(SCAN_REGEX "\n[ \t]+(constexpr )?(const )?auto&?&? [a-z_]+\\(")
