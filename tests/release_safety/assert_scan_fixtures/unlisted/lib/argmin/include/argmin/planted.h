#ifndef HPP_GUARD_ARGMIN_PLANTED_H
#define HPP_GUARD_ARGMIN_PLANTED_H

#include <cassert>

inline int planted_checked_index(int i, int n)
{
    static_assert(sizeof(int) >= 2);
    assert(i < n);
    return i;
}

#endif
