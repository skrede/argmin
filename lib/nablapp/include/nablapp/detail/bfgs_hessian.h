#ifndef HPP_GUARD_NABLAPP_DETAIL_BFGS_HESSIAN_H
#define HPP_GUARD_NABLAPP_DETAIL_BFGS_HESSIAN_H

// Direct-form BFGS Hessian approximation with Powell damping.
//
// Maintains B (the Hessian approximation itself, not its inverse).
// This is required by SQP because the QP subproblem uses B directly
// as the quadratic term: min 0.5 p^T B p + g^T p.
//
// An inverse-Hessian form (H = B^{-1}) would require O(n^3)
// inversion to recover B. This class avoids that cost.
//
// Reference: N&W eq. 8.19 (direct BFGS update);
//            N&W Procedure 18.2, eq. 18.22-18.24 (Powell damping);
//            N&W Section 18.4 (Hessian of augmented Lagrangian).

#include <Eigen/Core>

#include <cmath>

namespace nablapp::detail
{

template <typename Scalar = double>
class bfgs_hessian
{
public:
    explicit bfgs_hessian(int n = 0)
        : B_(Eigen::MatrixX<Scalar>::Identity(n, n))
    {}

    // Damped BFGS direct update.
    //
    // Given s = x_{k+1} - x_k and y = grad_L_{k+1} - grad_L_k,
    // update B to incorporate new curvature information.
    //
    // Powell damping (N&W Procedure 18.2, eq. 18.22-18.24):
    //   1. Compute Bs = B * s
    //   2. sBs = s^T B s
    //   3. sTy = s^T y
    //   4. If sTy >= 0.2 * sBs: r = y (no damping)
    //      Else: theta = 0.8 * sBs / (sBs - sTy),
    //            r = theta * y + (1 - theta) * Bs
    //   5. B_{k+1} = B - Bs Bs^T / sBs + r r^T / (s^T r)
    //
    // Reference: N&W eq. 8.19, Procedure 18.2.
    void update(const Eigen::VectorX<Scalar>& s,
                const Eigen::VectorX<Scalar>& y)
    {
        Eigen::VectorX<Scalar> Bs = B_ * s;
        Scalar sBs = s.dot(Bs);

        // Guard against degenerate step
        if(sBs < Scalar(1e-18))
            return;

        Scalar sTy = s.dot(y);

        // Powell damping
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

        Scalar sTr = s.dot(r);
        if(std::abs(sTr) < Scalar(1e-18))
            return;

        // Rank-2 direct BFGS update (N&W eq. 8.19)
        B_ += -Bs * Bs.transpose() / sBs + r * r.transpose() / sTr;
    }

    const Eigen::MatrixX<Scalar>& hessian() const { return B_; }

    Eigen::VectorX<Scalar> multiply(const Eigen::VectorX<Scalar>& v) const
    {
        return (B_ * v).eval();
    }

    void reset()
    {
        const int n = static_cast<int>(B_.rows());
        B_ = Eigen::MatrixX<Scalar>::Identity(n, n);
    }

    int dimension() const { return static_cast<int>(B_.rows()); }

private:
    Eigen::MatrixX<Scalar> B_;
};

}

#endif
