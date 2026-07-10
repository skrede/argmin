#ifndef HPP_GUARD_ARGMIN_DETAIL_INTERPOLATION_SYSTEM_H
#define HPP_GUARD_ARGMIN_DETAIL_INTERPOLATION_SYSTEM_H

// Powell's BMAT/ZMAT factored interpolation system for BOBYQA.
//
// Stores the inverse of the interpolation system's KKT matrix H in
// factored form: H's top-left m x m block is Z * Z^T (ZMAT), and
// the last n columns of H are stored directly (BMAT). This makes
// Lagrange polynomial evaluation, model updates, and denominator
// computation all O(m*n) instead of O(m*p^2) via SVD.
//
// The quadratic model Q(xopt + d) = f(xopt) + gopt^T d
//   + 0.5 d^T (HQ + sum_k pq[k] * xpt[k] * xpt[k]^T) d
// is stored as three components: GOPT (gradient at xopt), HQ (packed
// upper triangle of the explicit Hessian), and PQ (implicit second
// derivative coefficients, one per interpolation point).
//
// Reference: Powell, M. J. D. (2009) "The BOBYQA algorithm for bound
//            constrained optimization without derivatives",
//            DAMTP 2009/NA06, Sections 2-4.

#include <Eigen/Core>

#include <cmath>
#include <cstdint>

namespace argmin::detail
{

// Packed upper-triangle index for element (i, j) with i <= j.
//
//   hq_index(i, j, n) = i * n - i * (i + 1) / 2 + j
//
// Reference: Powell 2009, Section 2 (model representation).
constexpr int32_t hq_index(int32_t i, int32_t j, int32_t n)
{
    if(i > j)
    {
        auto tmp = i;
        i = j;
        j = tmp;
    }
    return i * n - i * (i + 1) / 2 + j;
}

// BMAT/ZMAT factored interpolation system.
//
// Template parameters:
//   Scalar -- floating-point type (default double)
//   N      -- problem dimension at compile time (default dynamic)
//
// For fixed N, all Eigen sizes are statically bounded to avoid heap
// allocation. For dynamic N, Eigen::Dynamic is used throughout.
//
// Reference: Powell 2009, Sections 2-4.
//   NLopt bobyqb_ comments at lines 1996-2012.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1996
template <typename Scalar = double, int N = Eigen::Dynamic>
struct interpolation_system
{
    // MaxM = max interpolation points (2*N+1 for fixed N).
    static constexpr int MaxM = (N == Eigen::Dynamic) ? Eigen::Dynamic : 2 * N + 1;

    // MaxMpN = max(m + n) = max rows in BMAT.
    static constexpr int MaxMpN = (N == Eigen::Dynamic) ? Eigen::Dynamic : 3 * N + 1;

    // MaxNptm = max(m - n - 1) = max columns in ZMAT. For NPT=2n+1, nptm=n.
    static constexpr int MaxNptm = (N == Eigen::Dynamic) ? Eigen::Dynamic : N;

    // MaxHQ = max packed upper-triangle size = n*(n+1)/2.
    static constexpr int MaxHQ = (N == Eigen::Dynamic) ? Eigen::Dynamic : N * (N + 1) / 2;

    // BMAT: (m+n) x n. Top m rows = interpolation point contributions;
    // bottom n rows = symmetric block.
    Eigen::Matrix<Scalar, MaxMpN, N> bmat;

    // ZMAT: m x nptm. Factor of top-left m x m block of H.
    Eigen::Matrix<Scalar, MaxM, MaxNptm> zmat;

    // Model gradient at xopt.
    Eigen::Vector<Scalar, N> gopt;

    // Packed upper triangle of explicit Hessian (n*(n+1)/2 entries).
    Eigen::Vector<Scalar, MaxHQ> hq;

    // Implicit second derivative coefficients, one per interpolation point.
    Eigen::Vector<Scalar, MaxM> pq;

    // Interpolation points relative to xbase (columns = points).
    Eigen::Matrix<Scalar, N, MaxM> xpt;

    // Base point for shifted coordinates.
    Eigen::Vector<Scalar, N> xbase;

    // Best point relative to xbase.
    Eigen::Vector<Scalar, N> xopt;

    // Function values at interpolation points.
    Eigen::Vector<Scalar, MaxM> fval;

    // Number of active interpolation points (NPT).
    int32_t m_points{0};

    // Index of best point in xpt/fval.
    int32_t kopt{0};
};

// Access element (i, j) of the explicit Hessian from packed HQ storage.
//
// Reference: Powell 2009, Section 2.
template <typename Scalar, int N>
Scalar hq_element(const interpolation_system<Scalar, N>& sys, int32_t i, int32_t j)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    return sys.hq[hq_index(i, j, n)];
}

// Bootstrap the interpolation system from an initial point.
//
// Creates the initial BMAT, ZMAT, GOPT, HQ, PQ, XPT, and FVAL arrays
// for NPT = 2n+1 points using coordinate perturbations of size rhobeg.
// This is the only path that builds the system from scratch; all
// subsequent updates use incremental rank-1/rank-2 operations.
//
// Parameters:
//   x0        -- initial point in original coordinates
//   rhobeg    -- initial trust-region radius (step size for perturbations)
//   lower     -- lower bounds in original coordinates
//   upper     -- upper bounds in original coordinates
//   eval_fn   -- callable(Vector) -> Scalar for objective evaluation
//
// The bounds are converted to scaled form sl[i] = lower[i] - x0[i],
// su[i] = upper[i] - x0[i] internally for bound-aware stepping.
//
// Reference: Powell 2009, Section 2.
//   Adapted from NLopt prelim_() lines 1710-1950.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1710
template <typename Scalar, int N, typename EvalFn>
interpolation_system<Scalar, N> bootstrap_interpolation_system(
    const Eigen::Vector<Scalar, N>& x0,
    Scalar rhobeg,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    EvalFn&& eval_fn)
{
    const int32_t n = static_cast<int32_t>(x0.size());
    const int32_t m = 2 * n + 1;
    const int32_t nptm = m - n - 1; // = n for default NPT=2n+1

    interpolation_system<Scalar, N> sys;
    sys.m_points = m;

    // Resize dynamic-size members if needed.
    if constexpr(N == Eigen::Dynamic)
    {
        sys.bmat.setZero(m + n, n);
        sys.zmat.setZero(m, nptm);
        sys.gopt.setZero(n);
        sys.hq.setZero(n * (n + 1) / 2);
        sys.pq.setZero(m);
        sys.xpt.setZero(n, m);
        sys.xbase.resize(n);
        sys.xopt.setZero(n);
        sys.fval.resize(m);
    }
    else
    {
        sys.bmat.setZero();
        sys.zmat.setZero();
        sys.gopt.setZero();
        sys.hq.setZero();
        sys.pq.setZero();
        sys.xpt.setZero();
        sys.xopt.setZero();
        sys.fval.setZero();
    }

    // xbase = x0 (the origin of the shifted coordinate system).
    sys.xbase = x0;

    // Scaled bounds: sl[i] = lower[i] - xbase[i], su[i] = upper[i] - xbase[i].
    Eigen::Vector<Scalar, N> sl = lower - x0;
    Eigen::Vector<Scalar, N> su = upper - x0;

    // Point 0 = origin (xbase itself).
    // xpt.col(0) is already zero.
    sys.fval[0] = eval_fn(x0);

    // Build points 1..n (positive perturbation along axis j)
    // and points n+1..2n (negative perturbation along axis j).
    //
    // Adapted from NLopt prelim_() lines 1826-1846.
    // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1826
    for(int32_t j = 0; j < n; ++j)
    {
        Scalar stepa = rhobeg;
        if(su[j] == Scalar(0))
            stepa = -stepa;
        sys.xpt(j, 1 + j) = stepa;

        // Evaluate point 1+j.
        Eigen::Vector<Scalar, N> pt = x0;
        pt[j] += stepa;
        sys.fval[1 + j] = eval_fn(pt);
    }

    for(int32_t j = 0; j < n; ++j)
    {
        Scalar stepa = sys.xpt(j, 1 + j);
        Scalar stepb = -rhobeg;
        if(sl[j] == Scalar(0))
            stepb = std::min(Scalar(2) * rhobeg, su[j]);
        if(su[j] == Scalar(0))
            stepb = std::max(Scalar(-2) * rhobeg, sl[j]);
        sys.xpt(j, 1 + n + j) = stepb;

        // Evaluate point n+1+j.
        Eigen::Vector<Scalar, N> pt = x0;
        pt[j] += stepb;
        sys.fval[1 + n + j] = eval_fn(pt);

        // Compute BMAT and ZMAT for this axis.
        //
        // Adapted from NLopt prelim_() lines 1894-1928.
        // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L1894
        Scalar fbeg = sys.fval[0];
        Scalar f_pos = sys.fval[1 + j];
        Scalar f_neg = sys.fval[1 + n + j];

        // Swap points if stepa and stepb have opposite signs and f_neg < f_pos,
        // so the better point sits at index 1+j. This must happen BEFORE the
        // BMAT/ZMAT assignments below: those encode the factored inverse of the
        // interpolation matrix at the final point positions, so assembling them
        // from the pre-swap coordinates breaks the Lagrange delta-property on
        // every swapped axis. Mirrors NLopt prelim_(), where the swap precedes
        // the BMAT/ZMAT writes. The two-sided stencil (hq/gopt) and the
        // symmetric factor terms are invariant under a consistent swap; only
        // bmat(1+n+j, j) = -0.5 / xpt(j, 1+j) reads the swapped position.
        if(stepa * stepb < Scalar(0) && f_neg < f_pos)
        {
            sys.fval[1 + n + j] = f_pos;
            sys.fval[1 + j] = f_neg;
            sys.xpt(j, 1 + j) = stepb;
            sys.xpt(j, 1 + n + j) = stepa;
        }

        // Gradient from two-sided stencil.
        Scalar diff = stepb - stepa;
        Scalar temp = (f_neg - fbeg) / stepb;
        int32_t ih = j * (j + 1) / 2 + j; // hq_index(j, j, n) for diagonal
        sys.hq[ih] = Scalar(2) * (temp - (f_pos - fbeg) / stepa) / diff;
        sys.gopt[j] = ((f_pos - fbeg) / stepa * stepb - temp * stepa) / diff;

        // BMAT initialization.
        // bmat(0, j) = -(1/stepa + 1/stepb)
        // bmat(1+j, j) = -1/xpt(j, n+1+j)     [note: NLopt uses different sign convention]
        // bmat(1+n+j, j) = -bmat(0,j) - bmat(1+j, j)
        //
        // Bottom block: bmat(m+j, j) = -0.5 * rhobeg^2  [from prelim_ L1900]
        sys.bmat(0, j) = -(stepa + stepb) / (stepa * stepb);
        sys.bmat(1 + n + j, j) = -Scalar(0.5) / sys.xpt(j, 1 + j);  // NLopt: nf-n row
        sys.bmat(1 + j, j) = -sys.bmat(0, j) - sys.bmat(1 + n + j, j);

        // ZMAT initialization.
        Scalar rhosq = rhobeg * rhobeg;
        sys.zmat(0, j) = std::sqrt(Scalar(2)) / (stepa * stepb);
        sys.zmat(1 + n + j, j) = std::sqrt(Scalar(0.5)) / rhosq;
        sys.zmat(1 + j, j) = -sys.zmat(0, j) - sys.zmat(1 + n + j, j);
    }

    // Find best point.
    sys.kopt = 0;
    for(int32_t k = 1; k < m; ++k)
    {
        if(sys.fval[k] < sys.fval[sys.kopt])
            sys.kopt = k;
    }
    sys.xopt = sys.xpt.col(sys.kopt);

    // Shift GOPT from xpt[0] (origin) to xopt.
    // gopt_new = gopt_old + HQ * xopt + sum_k PQ[k] * (xpt[k]^T xopt) * xpt[k]
    //
    // Reference: Powell 2009, Section 2.
    //   NLopt bobyqb_() lines 2110-2141 (L20 block).
    //   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2110
    if(sys.kopt != 0)
    {
        int32_t ih = 0;
        for(int32_t j = 0; j < n; ++j)
        {
            for(int32_t i = 0; i <= j; ++i)
            {
                if(i < j)
                    sys.gopt[j] += sys.hq[ih] * sys.xopt[i];
                sys.gopt[i] += sys.hq[ih] * sys.xopt[j];
                ++ih;
            }
        }
        for(int32_t k = 0; k < m; ++k)
        {
            auto xk = sys.xpt.col(k).head(n);
            Scalar temp = sys.pq[k] * xk.dot(sys.xopt);
            for(int32_t i = 0; i < n; ++i)
                sys.gopt[i] += temp * xk[i];
        }
    }

    return sys;
}

// Evaluate the interpolation model Q at xopt + d.
//
// Q(xopt + d) = fval[kopt] + gopt^T d + 0.5 d^T M d
// where M = HQ + sum_k pq[k] * xpt[k] * xpt[k]^T.
//
// The HQ contribution uses packed upper-triangle access; the PQ
// contribution uses dot products with xpt columns.
//
// Reference: Powell 2009, Section 2 (equation 2.2).
//   Adapted from NLopt bobyqb_() lines 2598-2621.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2598
template <typename Scalar, int N>
Scalar evaluate_interpolation_model(
    const interpolation_system<Scalar, N>& sys,
    const Eigen::Vector<Scalar, N>& d)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;

    // Linear term: gopt^T d.
    Scalar result = sys.gopt.dot(d);

    // HQ contribution: 0.5 d^T HQ d via packed upper triangle.
    int32_t ih = 0;
    for(int32_t j = 0; j < n; ++j)
    {
        for(int32_t i = 0; i <= j; ++i)
        {
            Scalar temp = d[i] * d[j];
            if(i == j)
                temp *= Scalar(0.5);
            result += sys.hq[ih] * temp;
            ++ih;
        }
    }

    // PQ contribution: 0.5 sum_k pq[k] * (xpt[k]^T d)^2.
    for(int32_t k = 0; k < m; ++k)
    {
        Scalar dot = sys.xpt.col(k).head(n).dot(d);
        result += Scalar(0.5) * sys.pq[k] * dot * dot;
    }

    return result;
}

// Gradient of the interpolation model at xopt + d.
//
// grad Q(xopt + d) = gopt + HQ * d + sum_k pq[k] * (xpt[k]^T d) * xpt[k]
//
// Reference: Powell 2009, Section 2.
//   Adapted from NLopt bobyqb_() model gradient computation.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2598
template <typename Scalar, int N>
Eigen::Vector<Scalar, N> model_gradient_at(
    const interpolation_system<Scalar, N>& sys,
    const Eigen::Vector<Scalar, N>& d)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;

    Eigen::Vector<Scalar, N> grad = sys.gopt;

    // HQ * d via packed upper triangle.
    int32_t ih = 0;
    for(int32_t j = 0; j < n; ++j)
    {
        for(int32_t i = 0; i <= j; ++i)
        {
            if(i == j)
            {
                grad[j] += sys.hq[ih] * d[j];
            }
            else
            {
                grad[i] += sys.hq[ih] * d[j];
                grad[j] += sys.hq[ih] * d[i];
            }
            ++ih;
        }
    }

    // PQ contribution: sum_k pq[k] * (xpt[k]^T d) * xpt[k].
    for(int32_t k = 0; k < m; ++k)
    {
        Scalar dot = sys.xpt.col(k).head(n).dot(d);
        grad += (sys.pq[k] * dot) * sys.xpt.col(k).head(n);
    }

    return grad;
}

// Result of VLAG/BETA computation.
template <typename Scalar, int N>
struct vlag_beta_result
{
    static constexpr int MaxMpN = interpolation_system<Scalar, N>::MaxMpN;

    // vlag has m+n entries: Lagrange function values at the trial point,
    // plus n entries for the bottom block.
    Eigen::Vector<Scalar, MaxMpN> vlag;

    // Beta parameter for the BMAT/ZMAT update denominator.
    Scalar beta{};
};

// Compute VLAG and BETA for a trial step d (relative to xopt).
//
// VLAG[k] = L_k(xopt + d) for k = 0..m-1 (Lagrange function values).
// VLAG[m+j] for j = 0..n-1 are the bottom-block contributions.
// BETA is the second denominator component: denom = vlag[knew]^2 + alpha * beta.
//
// Reference: Powell 2009, Sections 2 and 4.
//   Adapted from NLopt bobyqb_() lines 2394-2455.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2394
template <typename Scalar, int N>
vlag_beta_result<Scalar, N> compute_vlag_beta(
    const interpolation_system<Scalar, N>& sys,
    const Eigen::Vector<Scalar, N>& d)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;
    const int32_t nptm = m - n - 1;

    vlag_beta_result<Scalar, N> result;
    if constexpr(N == Eigen::Dynamic)
        result.vlag.setZero(m + n);
    else
        result.vlag.setZero();

    // w[k] = (xpt[k] . d) * (0.5*(xpt[k].d) + xpt[k].xopt)   for k=0..m-1
    // Also store suma = xpt[k] . d  in a temporary for later use.
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> w;
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> suma_vec;
    if constexpr(N == Eigen::Dynamic)
    {
        w.resize(m);
        suma_vec.resize(m);
    }

    for(int32_t k = 0; k < m; ++k)
    {
        auto xk = sys.xpt.col(k).head(n);
        Scalar suma = xk.dot(d);
        Scalar sumb = xk.dot(sys.xopt);
        w[k] = suma * (Scalar(0.5) * suma + sumb);
        suma_vec[k] = suma;

        // Linear contribution from BMAT to vlag: sum_j bmat[k,j] * d[j].
        result.vlag[k] = sys.bmat.row(k).head(n).dot(d);
    }

    // Quadratic contribution from ZMAT to vlag.
    // Also accumulate -sum(s^2) into beta.
    result.beta = Scalar(0);
    for(int32_t jj = 0; jj < nptm; ++jj)
    {
        Scalar s = Scalar(0);
        for(int32_t k = 0; k < m; ++k)
            s += sys.zmat(k, jj) * w[k];
        result.beta -= s * s;
        for(int32_t k = 0; k < m; ++k)
            result.vlag[k] += s * sys.zmat(k, jj);
    }

    // Bottom-block vlag entries and remaining beta terms.
    Scalar dsq = d.squaredNorm();
    Scalar bsum = Scalar(0);
    Scalar dx = d.dot(sys.xopt);
    Scalar xoptsq = sys.xopt.squaredNorm();

    for(int32_t j = 0; j < n; ++j)
    {
        // Sum over BMAT top block: sum_k w[k] * bmat[k, j].
        Scalar s = Scalar(0);
        for(int32_t k = 0; k < m; ++k)
            s += w[k] * sys.bmat(k, j);
        bsum += s * d[j];

        // Add symmetric bottom block: sum_i bmat[m+i, j] * d[i].
        for(int32_t i = 0; i < n; ++i)
            s += sys.bmat(m + i, j) * d[i];

        result.vlag[m + j] = s;
        bsum += s * d[j];
    }

    // Complete beta formula.
    // beta = dx^2 + dsq*(xoptsq + 2*dx + 0.5*dsq) + zmat_sum - bsum
    result.beta = dx * dx + dsq * (xoptsq + dx + dx + Scalar(0.5) * dsq)
                  + result.beta - bsum;

    // Add 1 to vlag[kopt] (the Lagrange value at the optimal point).
    result.vlag[sys.kopt] += Scalar(1);

    return result;
}

// Update BMAT and ZMAT after replacing interpolation point KNEW.
//
// Step 1: Apply Givens rotations to zero row KNEW of ZMAT (columns
//         1..nptm-1), concentrating all information into column 0.
// Step 2: Compute alpha and tau from the concentrated ZMAT.
// Step 3: Update ZMAT column 0.
// Step 4: Rank-2 BMAT update with symmetry enforcement on bottom block.
//
// Reference: Powell 2009, Section 4, equation (4.11).
//   Adapted from NLopt update_() lines 18-140.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L18
template <typename Scalar, int N>
void update_bmat_zmat(
    interpolation_system<Scalar, N>& sys,
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxMpN>& vlag,
    Scalar beta,
    Scalar denom,
    int32_t knew)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;
    const int32_t nptm = m - n - 1;

    // Compute threshold for treating ZMAT elements as zero.
    Scalar ztest = Scalar(0);
    for(int32_t k = 0; k < m; ++k)
    {
        for(int32_t j = 0; j < nptm; ++j)
        {
            Scalar absval = std::abs(sys.zmat(k, j));
            if(absval > ztest)
                ztest = absval;
        }
    }
    ztest *= Scalar(1e-20);

    // Step 1: Apply Givens rotations to zero row KNEW of ZMAT (cols 1..nptm-1).
    for(int32_t j = 1; j < nptm; ++j)
    {
        if(std::abs(sys.zmat(knew, j)) > ztest)
        {
            Scalar a = sys.zmat(knew, 0);
            Scalar b = sys.zmat(knew, j);
            Scalar r = std::sqrt(a * a + b * b);
            Scalar ca = a / r;
            Scalar cb = b / r;
            for(int32_t i = 0; i < m; ++i)
            {
                Scalar t = ca * sys.zmat(i, 0) + cb * sys.zmat(i, j);
                sys.zmat(i, j) = ca * sys.zmat(i, j) - cb * sys.zmat(i, 0);
                sys.zmat(i, 0) = t;
            }
        }
        sys.zmat(knew, j) = Scalar(0);
    }

    // Step 2: Compute w (first m components of KNEW-th column of H)
    // and alpha = w[knew].
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> w_vec;
    if constexpr(N == Eigen::Dynamic)
        w_vec.resize(m);
    for(int32_t i = 0; i < m; ++i)
        w_vec[i] = sys.zmat(knew, 0) * sys.zmat(i, 0);
    Scalar alpha = w_vec[knew];
    Scalar tau = vlag[knew];
    vlag[knew] -= Scalar(1);

    // Step 3: Update ZMAT column 0.
    Scalar sqrtd = std::sqrt(denom);
    Scalar tempb = sys.zmat(knew, 0) / sqrtd;
    Scalar tempa = tau / sqrtd;
    for(int32_t i = 0; i < m; ++i)
        sys.zmat(i, 0) = tempa * sys.zmat(i, 0) - tempb * vlag[i];

    // Step 4: Rank-2 BMAT update.
    // Build the full w vector (m+n entries) for the BMAT update.
    // First m entries = w_vec (from ZMAT), next n entries = bmat[knew, :].
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxMpN> w_full;
    if constexpr(N == Eigen::Dynamic)
        w_full.resize(m + n);
    for(int32_t i = 0; i < m; ++i)
        w_full[i] = w_vec[i];
    for(int32_t j = 0; j < n; ++j)
        w_full[m + j] = sys.bmat(knew, j);

    for(int32_t j = 0; j < n; ++j)
    {
        int32_t jp = m + j;
        Scalar wjp = w_full[jp];
        Scalar ta = (alpha * vlag[jp] - tau * wjp) / denom;
        Scalar tb = (-beta * wjp - tau * vlag[jp]) / denom;
        for(int32_t i = 0; i <= jp; ++i)
        {
            sys.bmat(i, j) += ta * vlag[i] + tb * w_full[i];
            // Enforce symmetry of the bottom n x n block.
            if(i >= m)
                sys.bmat(jp, i - m) = sys.bmat(i, j);
        }
    }
}

// Compute the denominator for the BMAT/ZMAT update.
//
// denom = vlag[knew]^2 + alpha * beta
//
// The caller should check that denom is sufficiently positive before
// calling update_bmat_zmat (e.g., denom > 0.5 * biglsq).
//
// Reference: Powell 2009, Section 4.
//   NLopt bobyqb_() lines 2462-2464.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2462
template <typename Scalar>
Scalar compute_denom(Scalar vlag_knew, Scalar alpha, Scalar beta)
{
    return vlag_knew * vlag_knew + alpha * beta;
}

// Compute Lagrange polynomial values at a query point via BMAT/ZMAT.
//
// Returns L_k(x_query) for k = 0..m-1, where x_query is given
// relative to xbase. This replaces the O(m*p^2) SVD-based approach
// with an O(m*n) matrix-vector multiply.
//
// Reference: Powell 2009, Sections 2-3.
//   Follows the same pattern as compute_vlag_beta but for an
//   arbitrary query point rather than xopt + d.
template <typename Scalar, int N>
Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> compute_lagrange_at(
    const interpolation_system<Scalar, N>& sys,
    const Eigen::Vector<Scalar, N>& x_query)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;
    const int32_t nptm = m - n - 1;

    // s = x_query - xopt (displacement from optimal point).
    Eigen::Vector<Scalar, N> s = x_query - sys.xopt;

    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> lagrange;
    if constexpr(N == Eigen::Dynamic)
        lagrange.resize(m);

    // w[k] = (xpt[k] . s) * (0.5*(xpt[k].s) + xpt[k].xopt).
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> w;
    if constexpr(N == Eigen::Dynamic)
        w.resize(m);

    for(int32_t k = 0; k < m; ++k)
    {
        auto xk = sys.xpt.col(k).head(n);
        Scalar dot_s = xk.dot(s);
        Scalar dot_xopt = xk.dot(sys.xopt);
        w[k] = dot_s * (Scalar(0.5) * dot_s + dot_xopt);

        // Linear contribution from BMAT.
        lagrange[k] = sys.bmat.row(k).head(n).dot(s);
    }

    // Quadratic contribution from ZMAT.
    for(int32_t jj = 0; jj < nptm; ++jj)
    {
        Scalar sum = Scalar(0);
        for(int32_t k = 0; k < m; ++k)
            sum += sys.zmat(k, jj) * w[k];
        for(int32_t k = 0; k < m; ++k)
            lagrange[k] += sum * sys.zmat(k, jj);
    }

    // The BMAT/ZMAT product gives H * phi(x), which equals the Lagrange
    // values minus the interpolation condition delta_{k,kopt}. Add the
    // correction so that L_kopt(xopt) = 1 (partition of unity).
    // This matches NLopt bobyqb_() line 2455: vlag[kopt] += one.
    lagrange[sys.kopt] += Scalar(1);

    return lagrange;
}

// Compute the gradient of L_knew (the KNEW-th Lagrange polynomial) at xopt.
//
// glag[i] = bmat[knew, i]                               (linear from BMAT)
// glag[i] += sum_k hcol[k] * (xpt[k] . xopt) * xpt[k,i]  (quadratic from ZMAT)
//
// where hcol[k] = sum_j zmat[knew, j] * zmat[k, j] is the KNEW-th column
// of H's top-left block.
//
// This provides the exact Lagrange gradient needed by ALTMOV for
// geometry improvement steps.
//
// Reference: Powell 2009, Section 6.
//   Adapted from NLopt altmov_() lines 825-862.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L825
template <typename Scalar, int N>
Eigen::Vector<Scalar, N> lagrange_gradient_bmat(
    const interpolation_system<Scalar, N>& sys,
    int32_t knew)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;
    const int32_t nptm = m - n - 1;

    // Compute hcol[k] = sum_j zmat[knew, j] * zmat[k, j].
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> hcol;
    if constexpr(N == Eigen::Dynamic)
        hcol.setZero(m);
    else
        hcol.setZero();

    for(int32_t jj = 0; jj < nptm; ++jj)
    {
        Scalar temp = sys.zmat(knew, jj);
        for(int32_t k = 0; k < m; ++k)
            hcol[k] += temp * sys.zmat(k, jj);
    }

    // glag[i] = bmat[knew, i] + sum_k hcol[k] * (xpt[k] . xopt) * xpt[k, i].
    Eigen::Vector<Scalar, N> glag;
    if constexpr(N == Eigen::Dynamic)
        glag.resize(n);
    for(int32_t i = 0; i < n; ++i)
        glag[i] = sys.bmat(knew, i);

    for(int32_t k = 0; k < m; ++k)
    {
        auto xk = sys.xpt.col(k).head(n);
        Scalar temp = hcol[k] * xk.dot(sys.xopt);
        for(int32_t i = 0; i < n; ++i)
            glag[i] += temp * xk[i];
    }

    return glag;
}

// Update the quadratic model after replacing point KNEW.
//
// After update_bmat_zmat has updated the interpolation system matrices,
// this function updates the model (GOPT, HQ, PQ) and the point data
// (XPT, FVAL, KOPT, XOPT) to reflect the new interpolation point.
//
// Parameters:
//   sys    -- interpolation system (modified in place)
//   vlag   -- VLAG vector (m+n entries, already modified by update_bmat_zmat)
//   new_x  -- new interpolation point relative to xbase
//   new_f  -- function value at the new point
//   knew   -- index of the replaced point
//   d      -- step vector (new_x - xopt, or equivalently xnew - xopt)
//
// Reference: Powell 2009, Section 4.
//   Adapted from NLopt bobyqb_() lines 2710-2822.
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2710
template <typename Scalar, int N>
void update_model_on_replacement(
    interpolation_system<Scalar, N>& sys,
    const Eigen::Vector<Scalar, N>& new_x,
    Scalar new_f,
    int32_t knew,
    const Eigen::Vector<Scalar, N>& d)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;
    const int32_t nptm = m - n - 1;

    // diff = actual_f - predicted_f = f_new - (fopt + Q(d))
    // where Q(d) = gopt^T d + 0.5 d^T H d is the model prediction.
    //
    // NLopt bobyqb_() line 2621: diff = f - fopt - vquad.
    // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2621
    Scalar vquad = evaluate_interpolation_model(sys, d);
    Scalar diff = new_f - sys.fval[sys.kopt] - vquad;

    // Absorb the old PQ[knew] into HQ (transfer implicit -> explicit).
    //
    // Adapted from NLopt bobyqb_() lines 2716-2727.
    // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2716
    Scalar pqold = sys.pq[knew];
    sys.pq[knew] = Scalar(0);
    int32_t ih = 0;
    for(int32_t i = 0; i < n; ++i)
    {
        Scalar temp = pqold * sys.xpt(i, knew);
        for(int32_t j = 0; j <= i; ++j)
        {
            sys.hq[ih] += temp * sys.xpt(j, knew);
            ++ih;
        }
    }

    // Distribute diff through ZMAT into PQ.
    //
    // Adapted from NLopt bobyqb_() lines 2728-2736.
    // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2728
    for(int32_t jj = 0; jj < nptm; ++jj)
    {
        Scalar temp = diff * sys.zmat(knew, jj);
        for(int32_t k = 0; k < m; ++k)
            sys.pq[k] += temp * sys.zmat(k, jj);
    }

    // Update XPT and FVAL BEFORE the GOPT w-vector computation.
    // NLopt bobyqb_() lines 2741-2746 set xpt[knew] = xnew before the
    // loop at lines 2749-2773 that reads xpt[k] for all k (including
    // k=knew). Using the old xpt[knew] produces incorrect GOPT.
    sys.fval[knew] = new_f;
    for(int32_t i = 0; i < n; ++i)
        sys.xpt(i, knew) = new_x[i];

    // Update GOPT.
    //
    // w[i] = bmat[knew, i] + sum_k suma_k * xpt[k, i]
    // where suma_k = sum_jj zmat[knew, jj] * zmat[k, jj] * (xpt[k] . xopt)
    //
    // Adapted from NLopt bobyqb_() lines 2741-2779.
    // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2741
    Eigen::Vector<Scalar, N> w_gopt;
    if constexpr(N == Eigen::Dynamic)
        w_gopt.resize(n);
    for(int32_t i = 0; i < n; ++i)
        w_gopt[i] = sys.bmat(knew, i);

    for(int32_t k = 0; k < m; ++k)
    {
        Scalar suma = Scalar(0);
        for(int32_t jj = 0; jj < nptm; ++jj)
            suma += sys.zmat(knew, jj) * sys.zmat(k, jj);

        auto xk = sys.xpt.col(k).head(n);
        Scalar sumb = xk.dot(sys.xopt);
        Scalar temp = suma * sumb;
        for(int32_t i = 0; i < n; ++i)
            w_gopt[i] += temp * xk[i];
    }

    for(int32_t i = 0; i < n; ++i)
        sys.gopt[i] += diff * w_gopt[i];

    // If the new point is best, update kopt, xopt, and shift gopt.
    //
    // Adapted from NLopt bobyqb_() lines 2783-2817.
    // https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2783
    if(new_f < sys.fval[sys.kopt])
    {
        // Shift gopt from xopt_old to xopt_new = xopt_old + d.
        // gopt_new = gopt_old + HQ * d + sum_k pq[k] * (xpt[k]^T d) * xpt[k]
        ih = 0;
        for(int32_t j = 0; j < n; ++j)
        {
            for(int32_t i = 0; i <= j; ++i)
            {
                if(i < j)
                    sys.gopt[j] += sys.hq[ih] * d[i];
                sys.gopt[i] += sys.hq[ih] * d[j];
                ++ih;
            }
        }
        for(int32_t k = 0; k < m; ++k)
        {
            auto xk = sys.xpt.col(k).head(n);
            Scalar temp = sys.pq[k] * xk.dot(d);
            for(int32_t i = 0; i < n; ++i)
                sys.gopt[i] += temp * xk[i];
        }

        sys.kopt = knew;
        sys.xopt = sys.xpt.col(knew).head(n);
    }
}

// Shift the base point XBASE so that XOPT becomes the origin.
//
// Severe cancellation occurs in the factored algebra when XOPT drifts far
// from XBASE. This re-expresses the entire interpolation system about
// xbase + xopt: BMAT (both the ZMAT-independent bottom-block terms and the
// ZMAT-dependent terms), the explicit Hessian HQ, and the interpolation
// points XPT are all re-centered, then xbase += xopt, xopt = 0, xoptsq = 0.
//
// The shift is an EXACT re-parameterization: the quadratic model Q and every
// Lagrange function L_k are numerically unchanged in absolute coordinates.
// The delta-property (L_k(x_j) = delta_kj) therefore survives the shift.
//
// The scaled bounds sl/su are NOT stored in the system; the caller subtracts
// the old xopt from its own sl/su (or, equivalently, keeps deriving them from
// lower_scaled - xbase, which is automatically consistent after the shift).
//
// Reference: Powell 2009, Section 5 (origin shift).
//   Ported statement-for-statement from NLopt bobyqb_() lines 2215-2316
//   (the L90 block).
//   https://github.com/stevengj/nlopt/blob/master/src/algs/bobyqa/bobyqa.c#L2215
template <typename Scalar, int N>
void shift_xbase(interpolation_system<Scalar, N>& sys)
{
    const int32_t n = static_cast<int32_t>(sys.xbase.size());
    const int32_t m = sys.m_points;
    const int32_t nptm = m - n - 1;

    const Scalar xoptsq = sys.xopt.squaredNorm();
    const Scalar fracsq = Scalar(0.25) * xoptsq;

    // Workspace vectors (fixed-size for compile-time N, resized otherwise).
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> wnpt;
    Eigen::Vector<Scalar, interpolation_system<Scalar, N>::MaxM> vlag;
    Eigen::Vector<Scalar, N> wloc;
    Eigen::Vector<Scalar, N> vloc;
    Eigen::Vector<Scalar, N> wcol;
    if constexpr(N == Eigen::Dynamic)
    {
        wnpt.resize(m);
        vlag.resize(m);
        wloc.resize(n);
        vloc.resize(n);
        wcol.resize(n);
    }

    // Changes to BMAT that do not depend on ZMAT (the symmetric bottom block).
    Scalar sumpq = Scalar(0);
    for(int32_t k = 0; k < m; ++k)
    {
        sumpq += sys.pq[k];
        Scalar sum = -Scalar(0.5) * xoptsq;
        for(int32_t i = 0; i < n; ++i)
            sum += sys.xpt(i, k) * sys.xopt[i];
        wnpt[k] = sum;
        Scalar temp = fracsq - Scalar(0.5) * sum;
        for(int32_t i = 0; i < n; ++i)
        {
            wloc[i] = sys.bmat(k, i);
            vloc[i] = sum * sys.xpt(i, k) + temp * sys.xopt[i];
            const int32_t ip = m + i;
            for(int32_t j = 0; j <= i; ++j)
                sys.bmat(ip, j) += wloc[i] * vloc[j] + vloc[i] * wloc[j];
        }
    }

    // Changes to BMAT that depend on ZMAT.
    for(int32_t jj = 0; jj < nptm; ++jj)
    {
        Scalar sumz = Scalar(0);
        Scalar sumw = Scalar(0);
        for(int32_t k = 0; k < m; ++k)
        {
            sumz += sys.zmat(k, jj);
            vlag[k] = wnpt[k] * sys.zmat(k, jj);
            sumw += vlag[k];
        }
        for(int32_t j = 0; j < n; ++j)
        {
            Scalar sum = (fracsq * sumz - Scalar(0.5) * sumw) * sys.xopt[j];
            for(int32_t k = 0; k < m; ++k)
                sum += vlag[k] * sys.xpt(j, k);
            wcol[j] = sum;
            for(int32_t k = 0; k < m; ++k)
                sys.bmat(k, j) += sum * sys.zmat(k, jj);
        }
        for(int32_t i = 0; i < n; ++i)
        {
            const int32_t ip = m + i;
            Scalar temp = wcol[i];
            for(int32_t j = 0; j <= i; ++j)
                sys.bmat(ip, j) += temp * wcol[j];
        }
    }

    // Complete the shift: revise the explicit Hessian HQ, re-center XPT,
    // and enforce symmetry of the BMAT bottom block.
    int32_t ih = 0;
    for(int32_t j = 0; j < n; ++j)
    {
        wcol[j] = -Scalar(0.5) * sumpq * sys.xopt[j];
        for(int32_t k = 0; k < m; ++k)
        {
            wcol[j] += sys.pq[k] * sys.xpt(j, k);
            sys.xpt(j, k) -= sys.xopt[j];
        }
        for(int32_t i = 0; i <= j; ++i)
        {
            sys.hq[ih] += wcol[i] * sys.xopt[j] + sys.xopt[i] * wcol[j];
            ++ih;
            sys.bmat(m + i, j) = sys.bmat(m + j, i);
        }
    }

    // xbase absorbs xopt; the best point becomes the new origin.
    sys.xbase += sys.xopt;
    sys.xopt.setZero();
}

}

#endif
