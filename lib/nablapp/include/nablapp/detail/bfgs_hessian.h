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

#include "nablapp/types.h"
#include "nablapp/options/bfgs_options.h"

#include <Eigen/Core>

#include <cmath>

namespace nablapp::detail
{

template <typename Scalar = double, int N = nablapp::dynamic_dimension>
class bfgs_hessian
{
public:
    explicit bfgs_hessian(int n = 0)
        : B_(Eigen::Matrix<Scalar, N, N>::Identity(n, n))
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
    void update(const Eigen::Vector<Scalar, N>& s,
                const Eigen::Vector<Scalar, N>& y,
                const bfgs_options& opts = {})
    {
        Eigen::Vector<Scalar, N> Bs = (B_ * s).eval();
        Scalar sBs = s.dot(Bs);

        // Guard against degenerate step
        if(sBs < Scalar(1e-18))
            return;

        Scalar sTy = s.dot(y);

        // Powell damping (N&W Procedure 18.2)
        const auto damp_thr = static_cast<Scalar>(opts.damping_threshold.value_or(0.2));
        const auto damp_fac = static_cast<Scalar>(opts.damping_factor.value_or(0.8));

        Eigen::Vector<Scalar, N> r;
        if(sTy >= damp_thr * sBs)
        {
            r = y;
        }
        else
        {
            Scalar theta = damp_fac * sBs / (sBs - sTy);
            r = (theta * y + (Scalar(1) - theta) * Bs).eval();
        }

        Scalar sTr = s.dot(r);
        if(std::abs(sTr) < Scalar(1e-18))
            return;

        // Rank-2 direct BFGS update (N&W eq. 8.19)
        B_ += -Bs * Bs.transpose() / sBs + r * r.transpose() / sTr;
    }

    const Eigen::Matrix<Scalar, N, N>& hessian() const { return B_; }

    Eigen::Vector<Scalar, N> multiply(const Eigen::Vector<Scalar, N>& v) const
    {
        return (B_ * v).eval();
    }

    void reset()
    {
        const int n = static_cast<int>(B_.rows());
        B_ = Eigen::Matrix<Scalar, N, N>::Identity(n, n);
    }

    int dimension() const { return static_cast<int>(B_.rows()); }

private:
    Eigen::Matrix<Scalar, N, N> B_;
};

}

#endif
