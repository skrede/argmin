#ifndef HPP_GUARD_ARGMIN_QP_DETAIL_RESET_ON_MOVE_FLAG_H
#define HPP_GUARD_ARGMIN_QP_DETAIL_RESET_ON_MOVE_FLAG_H

#include <utility>

namespace argmin::detail
{

// A moved-from owner must not keep reporting state whose storage moved away, so the flag reads
// false in the object it was moved from.
class reset_on_move_flag
{
public:
    explicit reset_on_move_flag(bool value) noexcept : value_(value) {}

    reset_on_move_flag(reset_on_move_flag&& other) noexcept : value_(std::exchange(other.value_, false)) {}

    reset_on_move_flag& operator=(reset_on_move_flag&& other) noexcept
    {
        value_ = std::exchange(other.value_, false);
        return *this;
    }

    reset_on_move_flag& operator=(bool value) noexcept
    {
        value_ = value;
        return *this;
    }

    operator bool() const noexcept { return value_; }

private:
    bool value_;
};

}

#endif
