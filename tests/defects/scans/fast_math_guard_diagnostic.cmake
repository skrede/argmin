set(SCAN_NAME "a library header diagnoses a fast-math build that removes the non-finite input guards")
set(SCAN_FILES "lib/argmin/include/argmin/*.h")
set(SCAN_REGEX "__FAST_MATH__|__FINITE_MATH_ONLY__")
set(SCAN_ABSENT TRUE)
