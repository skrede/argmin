#ifndef HPP_GUARD_NABLAPP_HESSIAN_BFGS_APPROXIMATION_H
#define HPP_GUARD_NABLAPP_HESSIAN_BFGS_APPROXIMATION_H

#include <Eigen/Core>

namespace nablapp
{

// Stub -- TDD RED phase.
template <typename Scalar = double>
class bfgs_approximation
{
public:
    explicit bfgs_approximation(int n) : H_(Eigen::MatrixX<Scalar>::Identity(n, n)) {}

    void update(const Eigen::VectorX<Scalar>&, const Eigen::VectorX<Scalar>&) {}

    Eigen::VectorX<Scalar> direction(const Eigen::VectorX<Scalar>& g) const
    {
        return Eigen::VectorX<Scalar>::Zero(g.size());
    }

    const Eigen::MatrixX<Scalar>& inverse_hessian() const { return H_; }
    void reset() { H_.setIdentity(); }
    int dimension() const { return static_cast<int>(H_.rows()); }

private:
    Eigen::MatrixX<Scalar> H_;
};

}

#endif
