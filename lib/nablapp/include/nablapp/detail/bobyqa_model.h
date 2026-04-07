#ifndef HPP_GUARD_NABLAPP_DETAIL_BOBYQA_MODEL_H
#define HPP_GUARD_NABLAPP_DETAIL_BOBYQA_MODEL_H

// Powell 2009 quadratic model representation for BOBYQA.
//
// Maintains the Lagrange interpolation polynomial Q(x) using the
// GOPT/HQ/PQ representation and the H-matrix inverse factorization
// via BMAT/ZMAT. This replaces the SVD-based quadratic_model with
// Powell's incremental O(n^2) update scheme.
//
// Reference: Powell, M. J. D. (2009) The BOBYQA algorithm for bound
//            constrained optimization without derivatives, DAMTP 2009/NA06.

#include "nablapp/types.h"
#include "nablapp/detail/bound_projection.h"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace nablapp::detail
{

// Quadratic interpolation model for BOBYQA (Powell 2009, Sec. 2).
//
// Template parameters:
//   Scalar -- floating-point type
//   N      -- problem dimension (compile-time or dynamic_dimension)
//   NPT    -- number of interpolation points (default 2N+1)
//
// The model Q(x_opt + d) = f_opt + gopt^T d + 0.5 d^T H_full d where
// H_full = HQ_matrix + sum_k pq[k] * xpt[k] * xpt[k]^T. HQ stores
// explicit second derivatives in packed upper-triangle form; PQ stores
// implicit Lagrange polynomial weights.
//
// The interpolation system inverse H is factored as:
//   Leading NPT x NPT block = ZMAT * ZMAT^T
//   Last N columns stored in BMAT (rows: NPT + N)
//
// Reference: Powell 2009, Sections 2 and 4.

template <typename Scalar = double,
          int N = nablapp::dynamic_dimension,
          int NPT = (N == nablapp::dynamic_dimension
                          ? nablapp::dynamic_dimension
                          : 2 * N + 1)>
struct bobyqa_model
{
    static_assert(N == nablapp::dynamic_dimension || NPT == 2 * N + 1,
                  "NPT must equal 2*N+1 when N is compile-time");

    // Derived dimension constants for fixed-size storage.
    static constexpr int hq_size =
        (N == nablapp::dynamic_dimension)
            ? nablapp::dynamic_dimension
            : N * (N + 1) / 2;

    static constexpr int npt_plus_n =
        (N == nablapp::dynamic_dimension || NPT == nablapp::dynamic_dimension)
            ? nablapp::dynamic_dimension
            : NPT + N;

    static constexpr int zmat_cols =
        (N == nablapp::dynamic_dimension || NPT == nablapp::dynamic_dimension)
            ? nablapp::dynamic_dimension
            : NPT - N - 1;

    // Model representation (Powell 2009, Sec. 2).
    Eigen::Vector<Scalar, N> gopt;                      // Gradient of Q at x_opt
    Eigen::Vector<Scalar, hq_size> hq;                  // Packed upper triangle of explicit 2nd derivatives
    Eigen::Vector<Scalar, NPT> pq;                      // Implicit 2nd derivative weights
    Eigen::Matrix<Scalar, npt_plus_n, N> bmat;          // Last N columns of H
    Eigen::Matrix<Scalar, NPT, zmat_cols> zmat;         // Z factor: H_leading = Z*Z^T
    Eigen::Matrix<Scalar, NPT, N> xpt;                  // Interpolation points relative to x_base
    Eigen::Vector<Scalar, NPT> fval;                    // Function values at interpolation points
    Eigen::Vector<Scalar, N> x_base;                    // Base point for numerical stability
    Eigen::Vector<Scalar, N> x_opt;                     // Current best point relative to x_base
    Scalar f_opt{};                                      // Best objective value
    std::uint16_t k_opt{0};                              // Index of best point in xpt/fval
    Scalar rho{};                                        // Current trust-region radius
    Scalar rho_beg{};                                    // Initial trust-region radius
    Scalar rho_end{};                                    // Final trust-region radius (stopping criterion)

    // Access element (i,j) of the symmetric matrix stored in hq (packed upper triangle).
    //
    // Index formula: for i <= j, k = i*(2*n - i - 1)/2 + j.
    //
    // Reference: Powell 2009, Sec. 2.
    Scalar& hq_element(int i, int j, int n)
    {
        if(i > j) std::swap(i, j);
        return hq[i * (2 * n - i - 1) / 2 + j];
    }

    Scalar hq_element(int i, int j, int n) const
    {
        if(i > j) std::swap(i, j);
        return hq[i * (2 * n - i - 1) / 2 + j];
    }

    // Evaluate Q(x_opt + d) - f_opt.
    //
    // Q(x_opt + d) = f_opt + gopt^T d + 0.5 d^T (HQ_mat + sum_k pq[k] * xpt[k] * xpt[k]^T) d
    //
    // Reference: Powell 2009, Sec. 2.
    Scalar evaluate(const Eigen::Vector<Scalar, N>& d) const
    {
        const int n = gopt.size();
        const int npt = pq.size();

        Scalar val = gopt.dot(d);

        // HQ contribution (packed symmetric).
        for(int i = 0; i < n; ++i)
        {
            Scalar sum = Scalar(0);
            for(int j = 0; j < n; ++j)
                sum += hq_element(i, j, n) * d[j];
            val += Scalar(0.5) * d[i] * sum;
        }

        // PQ contribution (implicit Hessian via interpolation points).
        for(int k = 0; k < npt; ++k)
        {
            Scalar dot = xpt.row(k).dot(d);
            val += Scalar(0.5) * pq[k] * dot * dot;
        }

        return val;
    }

    // Gradient of Q at x_opt + d.
    //
    // grad = gopt + HQ_mat * d + sum_k pq[k] * (xpt[k]^T d) * xpt[k]
    //
    // Reference: Powell 2009, Sec. 2.
    Eigen::Vector<Scalar, N> gradient(const Eigen::Vector<Scalar, N>& d) const
    {
        const int n = gopt.size();
        const int npt = pq.size();

        Eigen::Vector<Scalar, N> g = gopt;

        // HQ contribution.
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
                g[i] += hq_element(i, j, n) * d[j];
        }

        // PQ contribution.
        for(int k = 0; k < npt; ++k)
        {
            Scalar dot = xpt.row(k).dot(d);
            g.noalias() += (pq[k] * dot) * xpt.row(k).transpose();
        }

        return g;
    }

    // Compute Hessian-vector product H_full * v.
    //
    // H_full * v = HQ_mat * v + sum_k pq[k] * (xpt[k]^T v) * xpt[k]
    //
    // Reference: Powell 2009, Sec. 2.
    Eigen::Vector<Scalar, N> hessian_vector_product(const Eigen::Vector<Scalar, N>& v) const
    {
        const int n = gopt.size();
        const int npt = pq.size();

        Eigen::Vector<Scalar, N> result = Eigen::Vector<Scalar, N>::Zero(n);

        // HQ contribution.
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
                result[i] += hq_element(i, j, n) * v[j];
        }

        // PQ contribution.
        for(int k = 0; k < npt; ++k)
        {
            Scalar dot = xpt.row(k).dot(v);
            result.noalias() += (pq[k] * dot) * xpt.row(k).transpose();
        }

        return result;
    }

    // Initialize model: Powell's PRELIM subroutine.
    //
    // Sets up the initial interpolation set with 2N+1 points along
    // coordinate directions, accounting for bounds. Initializes XPT,
    // FVAL, BMAT, ZMAT, GOPT, HQ (zero), PQ.
    //
    // Per BOB-07: adjusts step sizes when bounds are tight, rescaling
    // to equalize initial steps across dimensions.
    //
    // Reference: Powell 2009, Sec. 2 (PRELIM).
    template <typename EvalFn>
    void initialize(const Eigen::Vector<Scalar, N>& x0,
                    const Eigen::Vector<Scalar, N>& lower,
                    const Eigen::Vector<Scalar, N>& upper,
                    Scalar rhobeg, Scalar rhoend,
                    EvalFn&& eval_fn)
    {
        const int n = x0.size();
        const int npt = 2 * n + 1;

        rho_beg = rhobeg;
        rho_end = rhoend;
        rho = rhobeg;

        // Resize dynamic storage if needed.
        resize_if_dynamic(n, npt);

        // Set base point and project to bounds.
        x_base = detail::project<Scalar, N>(x0, lower, upper);

        // Compute per-dimension step sizes (BOB-07: rescale for tight bounds).
        // Powell 2009, Sec. 2: step_i = min(rhobeg, (upper_i - lower_i) / 2).
        Eigen::Vector<Scalar, N> steps(n);
        for(int i = 0; i < n; ++i)
        {
            Scalar range = upper[i] - lower[i];
            steps[i] = std::min(rhobeg, range / Scalar(2));
        }

        // Build initial interpolation set: 2N+1 points along coordinate axes.
        // Point 0 is x_base (origin in shifted coords).
        // Points 1..N: x_base + step_i * e_i (or shifted if near bound).
        // Points N+1..2N: x_base - step_i * e_i (or shifted if near bound).
        //
        // Reference: Powell 2009, Sec. 2.
        xpt.setZero();
        for(int i = 0; i < n; ++i)
        {
            Scalar step = steps[i];

            // Positive direction point.
            Scalar pos = x_base[i] + step;
            if(pos > upper[i])
                pos = x_base[i] - step;
            xpt(1 + i, i) = pos - x_base[i];

            // Negative direction point.
            Scalar neg = x_base[i] - step;
            if(neg < lower[i])
                neg = x_base[i] + step;
            // Avoid duplicating the positive step point.
            if(std::abs((neg - x_base[i]) - xpt(1 + i, i)) < Scalar(1e-15) * step)
                neg = x_base[i] + Scalar(0.5) * step;
            xpt(1 + n + i, i) = neg - x_base[i];
        }

        // Evaluate function at all interpolation points.
        fval[0] = eval_fn(x_base);
        for(int k = 1; k < npt; ++k)
        {
            Eigen::Vector<Scalar, N> pt = x_base + xpt.row(k).transpose();
            pt = detail::project<Scalar, N>(pt, lower, upper);
            fval[k] = eval_fn(pt);
        }

        // Find best point.
        k_opt = 0;
        f_opt = fval[0];
        for(int k = 1; k < npt; ++k)
        {
            if(fval[k] < f_opt)
            {
                f_opt = fval[k];
                k_opt = static_cast<std::uint16_t>(k);
            }
        }
        x_opt = xpt.row(k_opt).transpose();

        // Initialize BMAT and ZMAT for the initial H matrix.
        //
        // For the 2N+1 coordinate-based initial set, Powell derives closed-form
        // expressions for the initial H. With XPT[0] = 0 and XPT[i]/XPT[N+i]
        // along coordinate axis i:
        //
        // Reference: Powell 2009, Sec. 2 (initial BMAT/ZMAT construction).
        initialize_h_matrix(n, npt, steps);

        // Initialize model coefficients from interpolation conditions.
        //
        // HQ starts at zero (no explicit second derivatives initially).
        // PQ and GOPT are derived from function values and the initial H.
        //
        // Reference: Powell 2009, Sec. 2.
        hq.setZero();
        initialize_model_coefficients(n, npt, steps);
    }

    // Shift x_base to x_base + x_opt, update XPT rows, GOPT, HQ.
    //
    // Called when x_opt grows large relative to rho for numerical stability.
    //
    // Reference: Powell 2009, Sec. 2 (base point shift).
    void shift_base(int n)
    {
        const int npt = pq.size();

        // Update GOPT to reflect the new base point.
        // New gradient = gradient(x_opt) where x_opt is about to become zero.
        Eigen::Vector<Scalar, N> g_at_xopt = gradient(x_opt);
        gopt = g_at_xopt;

        // Update HQ: the explicit Hessian is invariant under translation.
        // (No change needed for HQ since it represents the Hessian directly.)

        // Shift all interpolation points.
        for(int k = 0; k < npt; ++k)
            xpt.row(k) -= x_opt.transpose();

        // Update base point.
        x_base += x_opt;
        x_opt.setZero();
    }

    // Rebuild model coefficients (GOPT, HQ, PQ) from current XPT and FVAL.
    //
    // Computes the minimum Frobenius norm quadratic interpolant using SVD.
    // This is used when the incremental BMAT/ZMAT update becomes numerically
    // unstable. The model satisfies Q(xpt[k]) = fval[k] for all k.
    //
    // Also rebuilds BMAT and ZMAT from the interpolation system inverse.
    //
    // Reference: Powell 2009, Sec. 2 (model construction).
    void rebuild_model()
    {
        const int n = gopt.size();
        const int npt = pq.size();
        const int p = 1 + n + n * (n + 1) / 2;

        // Build polynomial basis matrix Phi (npt x p).
        // Basis: {1, s_1, ..., s_n, 0.5*s_1^2, s_1*s_2, ..., 0.5*s_n^2}
        // where s = xpt[k] - x_opt (shifted to be centered at x_opt).
        Eigen::MatrixXd phi(npt, p);

        for(int k = 0; k < npt; ++k)
        {
            Eigen::VectorXd s = (xpt.row(k).transpose() - x_opt).template cast<double>();

            phi(k, 0) = 1.0;

            for(int j = 0; j < n; ++j)
                phi(k, 1 + j) = s[j];

            int idx = 1 + n;
            for(int j = 0; j < n; ++j)
            {
                for(int l = j; l < n; ++l)
                {
                    double factor = (j == l) ? 0.5 : 1.0;
                    phi(k, idx) = factor * s[j] * s[l];
                    ++idx;
                }
            }
        }

        // Solve min ||theta||_2 s.t. Phi * theta = fval via SVD.
        Eigen::VectorXd fv = fval.template cast<double>();
        auto svd = phi.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        svd.setThreshold(1e-12);
        Eigen::VectorXd theta = svd.solve(fv);

        // Extract model coefficients from theta.
        // theta[0] = constant (f_opt, ignored -- we track f_opt separately).
        // theta[1..n] = gradient at x_opt.
        for(int j = 0; j < n; ++j)
            gopt[j] = static_cast<Scalar>(theta[1 + j]);

        // Reconstruct symmetric H from upper-triangular entries -> HQ.
        hq.setZero();
        int idx = 1 + n;
        for(int j = 0; j < n; ++j)
        {
            for(int l = j; l < n; ++l)
            {
                Scalar val = static_cast<Scalar>(theta[idx]);
                hq_element(j, l, n) = val;
                ++idx;
            }
        }

        // Zero PQ since the SVD model absorbs everything into HQ.
        pq.setZero();

        // Rebuild BMAT/ZMAT for the current point set.
        //
        // The H matrix is the inverse of the interpolation system
        // Omega. For general point sets, we compute Omega and invert it.
        //
        // Omega = [0_npt_npt, Phi_linear^T; Phi_linear, 0_nn]
        // where Phi_linear is the linear+constant part (npt x (1+n)).
        //
        // For simplicity, rebuild using the coordinate-based approach
        // that initialize_h_matrix uses, applied to the actual point set.
        // Since we use SVD for the model, BMAT/ZMAT are only needed for
        // compute_vlag and point replacement selection. For correctness,
        // we reconstruct them from the interpolation system.
        rebuild_h_matrix();
    }

    // Rebuild BMAT/ZMAT from the current XPT.
    //
    // Constructs the full interpolation system matrix Omega, inverts it,
    // and extracts BMAT and ZMAT from the inverse.
    //
    // The Omega matrix has structure (Powell 2009, Sec. 2):
    //   Omega = [ zeta   Phi^T ]
    //           [ Phi    0     ]
    // where zeta[j,k] = 0.5 * (xpt[j]^T xpt[k])^2 for j,k < npt
    // and Phi[j,i] = [1, xpt[j]^T] for the last 1+n rows/columns.
    //
    // The inverse H = Omega^{-1} gives:
    //   H[0:npt, 0:npt] = ZMAT * ZMAT^T (leading block)
    //   H[:, npt:] = BMAT (last n columns, all rows)
    //
    // Reference: Powell 2009, Sec. 2.
    void rebuild_h_matrix()
    {
        const int n = gopt.size();
        const int npt = pq.size();
        const int m = npt + n + 1;

        // Build Omega matrix.
        Eigen::MatrixXd omega = Eigen::MatrixXd::Zero(m, m);

        // Upper-left block: zeta[j,k] = 0.5 * (xpt[j]^T xpt[k])^2.
        for(int j = 0; j < npt; ++j)
        {
            for(int k = j; k < npt; ++k)
            {
                double dot = xpt.row(j).dot(xpt.row(k));
                double val = 0.5 * dot * dot;
                omega(j, k) = val;
                omega(k, j) = val;
            }
        }

        // Off-diagonal blocks: Phi = [1, xpt^T].
        // Lower-left: Phi rows in rows npt..npt+n, columns 0..npt-1.
        // Upper-right: Phi^T in rows 0..npt-1, columns npt..npt+n.
        for(int k = 0; k < npt; ++k)
        {
            omega(npt, k) = 1.0;
            omega(k, npt) = 1.0;
            for(int i = 0; i < n; ++i)
            {
                omega(npt + 1 + i, k) = static_cast<double>(xpt(k, i));
                omega(k, npt + 1 + i) = static_cast<double>(xpt(k, i));
            }
        }

        // Invert Omega.
        auto svd = omega.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        svd.setThreshold(1e-12);
        Eigen::MatrixXd H_full = svd.solve(Eigen::MatrixXd::Identity(m, m));

        // Extract BMAT: rows 0..npt+n-1, columns correspond to the
        // last n entries. BMAT[k, i] = H[k, npt + 1 + i] for k < npt.
        // BMAT[npt + i, j] = H[npt + 1 + i, npt + 1 + j] for the sigma block.
        bmat.setZero();
        for(int k = 0; k < npt; ++k)
        {
            for(int i = 0; i < n; ++i)
                bmat(k, i) = static_cast<Scalar>(H_full(k, npt + 1 + i));
        }
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
                bmat(npt + i, j) = static_cast<Scalar>(H_full(npt + 1 + i, npt + 1 + j));
        }

        // Extract ZMAT: factor the leading npt x npt block of H as Z * Z^T.
        // H_leading = H[0:npt, 0:npt].
        // Use eigendecomposition: H_leading = U * D * U^T, then Z = U * sqrt(D).
        // Only keep columns with positive eigenvalues (rank = npt - n - 1).
        Eigen::MatrixXd H_leading(npt, npt);
        for(int j = 0; j < npt; ++j)
            for(int k = 0; k < npt; ++k)
                H_leading(j, k) = H_full(j, k);

        auto svd_leading = H_leading.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        const int zmat_c = npt - n - 1;

        zmat.setZero();
        auto& sv = svd_leading.singularValues();
        for(int j = 0; j < zmat_c && j < sv.size(); ++j)
        {
            if(sv[j] < 1e-15) continue;
            double sqrtd = std::sqrt(sv[j]);
            for(int k = 0; k < npt; ++k)
                zmat(k, j) = static_cast<Scalar>(svd_leading.matrixU()(k, j) * sqrtd);
        }
    }

    // Progressive rho contraction per Powell 2009 Sec. 5 and BOB-02.
    //
    // When trust region iterations exhaust without sufficient improvement,
    // contract rho. Schedule: rho_new = max(rho * ratio, rho_end) where
    // ratio depends on rho/rho_beg (typically 0.1 for large rho, sqrt
    // for small). If rho_new <= rho_end * 1.5, set rho_new = rho_end.
    //
    // Reference: Powell 2009, Sec. 5.
    void update_rho()
    {
        Scalar ratio;
        Scalar rho_ratio = rho / rho_beg;

        // Large rho: aggressive contraction.
        // Small rho: gentler contraction (square root schedule).
        // Reference: Powell 2009, Sec. 5.
        if(rho_ratio > Scalar(0.1))
            ratio = Scalar(0.1);
        else if(rho_ratio > Scalar(0.01))
            ratio = std::sqrt(rho_ratio);
        else
            ratio = Scalar(0.1);

        Scalar rho_new = std::max(rho * ratio, rho_end);

        // Snap to rho_end if close enough.
        if(rho_new <= rho_end * Scalar(1.5))
            rho_new = rho_end;

        rho = rho_new;
    }

private:
    // Resize dynamic-size members when N is dynamic_dimension.
    void resize_if_dynamic(int n, int npt)
    {
        if constexpr(N == nablapp::dynamic_dimension)
        {
            const int hq_sz = n * (n + 1) / 2;
            const int zmat_c = npt - n - 1;

            gopt.resize(n);
            hq.resize(hq_sz);
            pq.resize(npt);
            bmat.resize(npt + n, n);
            zmat.resize(npt, zmat_c);
            xpt.resize(npt, n);
            fval.resize(npt);
            x_base.resize(n);
            x_opt.resize(n);
        }
    }

    // Initialize the H-matrix factorization (BMAT/ZMAT) for the initial
    // coordinate-based interpolation set.
    //
    // For the symmetric 2N+1 initialization, each coordinate direction has
    // two points x_base +/- step_i * e_i. Powell derives the initial H
    // analytically.
    //
    // Reference: Powell 2009, Sec. 2.
    void initialize_h_matrix(int n, int npt, const Eigen::Vector<Scalar, N>& steps)
    {
        bmat.setZero();
        zmat.setZero();

        // For each coordinate i, the positive step xpt[1+i,i] and negative
        // step xpt[1+n+i,i] define two points. The initial H rows/cols
        // corresponding to these points are derived from the step sizes.
        //
        // BMAT row 0 (base point) gets contributions from all coordinates.
        // BMAT rows 1+i and 1+n+i get coordinate-specific entries.
        //
        // Reference: Powell 2009, Sec. 2 (PRELIM construction).
        for(int i = 0; i < n; ++i)
        {
            Scalar sp = xpt(1 + i, i);        // Positive step along axis i
            Scalar sn = xpt(1 + n + i, i);    // Negative step along axis i
            Scalar denom = sp * sn;

            if(std::abs(denom) < std::numeric_limits<Scalar>::epsilon())
                continue;

            // BMAT entries for the base point row 0.
            bmat(0, i) += -(sp + sn) / denom;

            // BMAT entries for the positive step point row 1+i.
            bmat(1 + i, i) = -sn / denom;

            // BMAT entries for the negative step point row 1+n+i.
            bmat(1 + n + i, i) = -sp / denom;

            // BMAT entries for the lower block (rows npt..npt+n-1).
            // These correspond to the "sigma" part of H.
            bmat(npt + i, i) = -Scalar(1) / sp - Scalar(1) / sn;

            // ZMAT: initial Z factor. Column i corresponds to the pair of
            // points along coordinate i.
            // Z[0, i] = sqrt(2) / (sp * sn) -- contribution from base point.
            // Z[1+i, i] and Z[1+n+i, i] from the two step points.
            Scalar recip_sp = Scalar(1) / sp;
            Scalar recip_sn = Scalar(1) / sn;
            Scalar z_base = std::sqrt(Scalar(2)) / std::abs(denom);

            zmat(0, i) = z_base;
            zmat(1 + i, i) = recip_sp / std::abs(sp) * std::sqrt(Scalar(0.5));
            zmat(1 + n + i, i) = recip_sn / std::abs(sn) * std::sqrt(Scalar(0.5));
        }
    }

    // Compute initial GOPT and PQ from function values and the initial H.
    //
    // For the 2N+1 coordinate-based initial set, the gradient and PQ
    // coefficients follow directly from finite differences along each axis.
    //
    // Reference: Powell 2009, Sec. 2 (PRELIM coefficient initialization).
    void initialize_model_coefficients(int n, int npt, const Eigen::Vector<Scalar, N>& steps)
    {
        gopt.setZero();
        pq.setZero();

        Scalar f0 = fval[0];

        for(int i = 0; i < n; ++i)
        {
            Scalar sp = xpt(1 + i, i);        // Positive step
            Scalar sn = xpt(1 + n + i, i);    // Negative step
            Scalar fp = fval[1 + i];           // f at positive step
            Scalar fn = fval[1 + n + i];       // f at negative step
            Scalar denom = sp * sn;

            if(std::abs(denom) < std::numeric_limits<Scalar>::epsilon())
                continue;

            // Gradient component via finite differences.
            // g_i = (fp * sn^2 - fn * sp^2 - f0 * (sn^2 - sp^2)) / (sp * sn * (sp - sn))
            Scalar sp2 = sp * sp;
            Scalar sn2 = sn * sn;
            Scalar diff = sp - sn;

            if(std::abs(diff) < std::numeric_limits<Scalar>::epsilon())
            {
                // Symmetric case: simple central difference.
                gopt[i] = (fp - fn) / (Scalar(2) * sp);
            }
            else
            {
                // Gradient via Cramer's rule on the interpolation system.
                // The denominator is sp*sn*(sn - sp) = -sp*sn*(sp - sn).
                gopt[i] = -(fp * sn2 - fn * sp2 - f0 * (sn2 - sp2)) / (denom * diff);
            }

            // PQ coefficients for points 1+i and 1+n+i.
            // These represent the implicit second derivative contributions.
            // From the interpolation conditions and the initial H matrix.
            //
            // Reference: Powell 2009, Sec. 2.
            Scalar second_deriv = Scalar(2) * (fp * sn - fn * sp - f0 * (sn - sp)) / (denom * diff);
            pq[1 + i] = (fn - f0 - sn * gopt[i]) / (sn * sn);
            pq[1 + n + i] = (fp - f0 - sp * gopt[i]) / (sp * sp);
        }

        // Adjust GOPT to be at x_opt instead of x_base (if k_opt != 0).
        //
        // gopt currently holds the gradient at x_base (d=0). The gradient
        // at x_opt is gopt_base + H * x_opt, which is exactly what
        // hessian_vector_product(x_opt) adds.
        //
        // Reference: Powell 2009, Sec. 2.
        if(k_opt != 0)
        {
            Eigen::Vector<Scalar, N> hv = hessian_vector_product(x_opt);
            gopt += hv;
        }
    }
};

}

#endif
