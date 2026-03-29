#ifndef HPP_GUARD_NABLAPP_HESSIAN_LBFGS_HISTORY_H
#define HPP_GUARD_NABLAPP_HESSIAN_LBFGS_HISTORY_H

#include <Eigen/Core>

namespace nablapp
{

// Stub -- TDD RED phase.
template <typename Scalar = double>
class lbfgs_history
{
public:
    explicit lbfgs_history(int m = 10) : capacity_(m) {}

    void push(const Eigen::VectorX<Scalar>&, const Eigen::VectorX<Scalar>&) {}

    Eigen::VectorX<Scalar> two_loop_recursion(const Eigen::VectorX<Scalar>& g) const
    {
        return Eigen::VectorX<Scalar>::Zero(g.size());
    }

    void reset() {}
    int size() const { return 0; }
    int capacity() const { return capacity_; }

private:
    int capacity_;
};

}

#endif
