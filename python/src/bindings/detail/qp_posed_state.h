#ifndef HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_QP_POSED_STATE_H
#define HPP_GUARD_ARGMIN_PYTHON_BINDINGS_DETAIL_QP_POSED_STATE_H

#include "bindings/detail/errors.h"

#include <string>
#include <string_view>

namespace argmin::python
{

// The library reports an unposed resolve through the same generic value it uses
// for several other precondition failures, so the distinction is not
// recoverable from the return value and has to be tracked here.
class posed_state
{
public:
    void pose(int n, int m)
    {
        n_ = n;
        m_ = m;
        posed_ = true;
    }

    void require_posed(std::string_view operation) const
    {
        if(posed_)
            return;
        raise_argmin_error(error_kind::invalid_state,
                           std::string(operation)
                               + " requires a preceding solve to pose the problem");
    }

    [[nodiscard]] int posed_rows() const { return n_; }
    [[nodiscard]] int posed_constraints() const { return m_; }

private:
    int n_{0};
    int m_{0};
    bool posed_{false};
};

}

#endif
