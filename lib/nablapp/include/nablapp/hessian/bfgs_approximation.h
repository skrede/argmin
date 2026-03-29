#ifndef HPP_GUARD_NABLAPP_HESSIAN_BFGS_APPROXIMATION_H
#define HPP_GUARD_NABLAPP_HESSIAN_BFGS_APPROXIMATION_H

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <utility>

namespace nablapp
{

// Dense BFGS inverse Hessian approximation with Powell damping.
//
// Maintains a symmetric positive-definite approximation H to the
// inverse Hessian. update(s, y) applies the rank-2 BFGS update
// with Powell's damping strategy to guarantee H remains PD even
// on nonconvex problems.
//
// Reference: N&W eq. 8.16 (BFGS inverse update),
//            N&W Procedure 18.2, eq. 18.22-18.24 (Powell damping).

template <typename Scalar = double>
class bfgs_approximation
{
public:
    explicit bfgs_approximation(int n) : H_(Eigen::MatrixX<Scalar>::Identity(n, n)) {}

    // Apply Powell-damped BFGS inverse Hessian update.
    //
    // Given step s = x_{k+1} - x_k and gradient change y = g_{k+1} - g_k,
    // update H to incorporate curvature information. Powell damping ensures
    // positive definiteness even when s^T y is small or negative.
    //
    // N&W eq. 8.16 with Procedure 18.2 damping.
    void update(const Eigen::VectorX<Scalar>& s, const Eigen::VectorX<Scalar>& y)
    {
        const int n = static_cast<int>(H_.rows());

        // B = H^{-1}, so B*s is obtained via LDLT factorization of H.
        // N&W Procedure 18.2: compute Bs for damping check.
        Eigen::VectorX<Scalar> Bs = H_.ldlt().solve(s);
        Scalar sTy = s.dot(y);
        Scalar sBs = s.dot(Bs);

        // Powell damping (N&W eq. 18.22-18.24):
        // If s^T y >= 0.2 * s^T B s, no damping needed.
        // Otherwise, interpolate r = theta*y + (1-theta)*Bs to ensure
        // sufficient curvature for a well-conditioned update.
        Eigen::VectorX<Scalar> r;
        if(sTy >= Scalar(0.2) * sBs)
        {
            r = y;
        }
        else
        {
            Scalar theta = Scalar(0.8) * sBs / (sBs - sTy);
            r = theta * y + (Scalar(1) - theta) * Bs;
        }

        // BFGS inverse update (N&W eq. 8.16) with r replacing y:
        // H_{k+1} = V^T H_k V + rho s s^T
        // where V = I - rho r s^T (note: transposed from some formulations),
        // and rho = 1 / (s^T r).
        Scalar rho = Scalar(1) / s.dot(r);
        Eigen::MatrixX<Scalar> I = Eigen::MatrixX<Scalar>::Identity(n, n);
        Eigen::MatrixX<Scalar> V = I - rho * s * r.transpose();
        Eigen::MatrixX<Scalar> H_new = V * H_ * V.transpose() + rho * s * s.transpose();
        H_ = std::move(H_new);
    }

    // Return search direction d = -H * g.
    Eigen::VectorX<Scalar> direction(const Eigen::VectorX<Scalar>& g) const
    {
        return -H_ * g;
    }

    const Eigen::MatrixX<Scalar>& inverse_hessian() const { return H_; }

    void reset()
    {
        const int n = static_cast<int>(H_.rows());
        H_ = Eigen::MatrixX<Scalar>::Identity(n, n);
    }

    int dimension() const { return static_cast<int>(H_.rows()); }

private:
    Eigen::MatrixX<Scalar> H_;
};

}

#endif
