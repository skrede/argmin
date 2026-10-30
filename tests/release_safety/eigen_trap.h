#ifndef HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_EIGEN_TRAP_H
#define HPP_GUARD_ARGMIN_TESTS_RELEASE_SAFETY_EIGEN_TRAP_H

#include <stdexcept>

// Eigen only defines eigen_assert when none exists, so this must precede every Eigen include to stay armed under NDEBUG.
#ifndef eigen_assert
#define eigen_assert(cond)                                                    \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
            throw std::runtime_error("eigen_assert: " #cond);                 \
    } while(false)
#endif

#endif
