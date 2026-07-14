#ifndef HPP_GUARD_ARGMIN_DETAIL_ACTIVE_SET_QP_H
#define HPP_GUARD_ARGMIN_DETAIL_ACTIVE_SET_QP_H

// Dense active-set QP solver.
//
// Solves: min 0.5 * x^T G x + d^T x
//         s.t. A_eq x = b_eq           (equality)
//              A_ineq x >= b_ineq      (inequality)
//
// Reference: N&W Section 16.1, Algorithm 16.1, pp. 460-463 (active-set method
//            for convex QP; phase-1 feasibility invariant on x0)
//            N&W Section 16.2, eq. 16.16-16.19 (null-space method)
//            N&W eq. 16.29 (blocking step length)
//            N&W Section 16.5, Lemma 16.5 (indefinite QP extensions; minimum-
//            norm under-determined solution via thin QR)

#include "argmin/detail/givens_qr_update.h"
#include "argmin/types.h"
#include "argmin/options/qp_options.h"

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace argmin::detail
{

enum class qp_status { optimal, infeasible, max_iterations, indefinite_hessian };

// MaxM bounds the multiplier storage without fixing its runtime length:
// box-constrained solves return m_eq + m_ineq + 2n multipliers (bound rows
// included), so a solver with compile-time constraint bounds sizes lambda
// at runtime up to MaxM = max_total_constraints with inline storage. The
// default MaxM = M keeps the plain fixed / dynamic shapes unchanged.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int M = argmin::dynamic_dimension, int MaxM = M>
struct qp_result
{
    Eigen::Vector<Scalar, N> x;
    Eigen::Matrix<Scalar, M, 1, 0, MaxM, 1> lambda;
    qp_status status{qp_status::optimal};
    int iterations{0};

    // Set by kraft_lsq_qp_recovery_solver when the inner cascade returned
    // infeasible/rank-deficient and Kraft 1988 §3.4 augmented-QP recovery
    // produced the step. Range [0, 1]: 0 means recovery was unnecessary
    // (or the direct path succeeded); positive values give the slack
    // variable s at the augmented optimum, where s=1 corresponds to full
    // constraint relaxation. Direct path always returns 0.
    Scalar relaxation_factor{Scalar(0)};
};

// Determine initial working set from feasible point x0.
// Equality constraints (indices 0..n_eq-1) are always candidates.
// Inequality constraints are candidates if |a_i^T x0 - b_i| < tolerance.
//
// A candidate is admitted only if its gradient row is linearly independent
// of the rows already admitted: the null-space / multiplier machinery
// (N&W Algorithm 16.3) requires A_W to have full row rank. Admitting a
// linearly dependent row (e.g. constraints x1 >= 0 and 2*x1 >= 0 both active
// at the same vertex) yields a rank-deficient A_W whose unpivoted QR solve
// produces NaN multipliers that then compare as "no negative multiplier",
// so the solver returns the start vertex mislabeled optimal. Candidates are
// scanned in ascending index (equalities first) so the retained set is the
// lowest-index independent basis, which is consistent with Bland's anti-
// cycling rule applied elsewhere.
//
// Rank is tested with Eigen's ColPivHouseholderQR at its default rank
// threshold (machine epsilon scaled by the matrix dimension) -- the library-
// standard relative rank tolerance, not a hand-tuned absolute constant.
//
// Reference: N&W Section 16.4, p. 460; N&W Algorithm 16.3 (full-row-rank
//            working-set prerequisite).
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
std::vector<int> initial_working_set(
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const Eigen::Vector<Scalar, M>& b_full,
    int n_eq,
    Scalar tolerance)
{
    const int m = static_cast<int>(A_full.rows());
    const int n = static_cast<int>(A_full.cols());

    std::vector<int> candidates;
    candidates.reserve(static_cast<std::size_t>(m));
    for(int i = 0; i < n_eq; ++i)
        candidates.push_back(i);
    for(int i = n_eq; i < m; ++i)
    {
        Scalar residual = A_full.row(i).dot(x0) - b_full[i];
        if(std::abs(residual) < tolerance)
            candidates.push_back(i);
    }

    std::vector<int> W;
    W.reserve(candidates.size());
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_W(0, n);
    int rank = 0;
    for(int idx : candidates)
    {
        if(rank >= n) break;  // at most n independent rows exist
        Eigen::Matrix<Scalar, Eigen::Dynamic, N> trial(A_W.rows() + 1, n);
        if(A_W.rows() > 0) trial.topRows(A_W.rows()) = A_W;
        trial.row(A_W.rows()) = A_full.row(idx);
        Eigen::ColPivHouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, N>> qr(trial);
        const int new_rank = static_cast<int>(qr.rank());
        if(new_rank > rank)
        {
            A_W = std::move(trial);
            rank = new_rank;
            W.push_back(idx);
        }
    }
    return W;
}

// Solve equality-constrained QP subproblem via null-space method.
//
// min 0.5 p^T G p + g_k^T p   s.t.  A_W p = 0
//
// Uses HouseholderQR of A_W^T for null-space computation (working set is
// full rank by construction, so rank-revealing is unnecessary).
// Uses LLT for reduced Hessian Z^T G Z solve (positive definite by construction).
//
// Returns (step p, multipliers lambda_W for working set constraints).
//
// Reference: N&W Section 16.2, eq. 16.16-16.19.
template <typename Scalar, int N = argmin::dynamic_dimension, int Mw = argmin::dynamic_dimension>
std::pair<Eigen::Vector<Scalar, N>, Eigen::Vector<Scalar, Mw>> solve_equality_qp(
    const Eigen::Matrix<Scalar, N, N>& G,
    const Eigen::Vector<Scalar, N>& g_k,
    const Eigen::Matrix<Scalar, Mw, N>& A_W,
    bool* reduced_hessian_ok = nullptr)
{
    const int n = static_cast<int>(G.rows());
    const int m = static_cast<int>(A_W.rows());
    if(reduced_hessian_ok) *reduced_hessian_ok = true;

    // No active constraints: unconstrained sub-step
    if(m == 0)
    {
        Eigen::LLT<Eigen::Matrix<Scalar, N, N>> llt(G);
        if(reduced_hessian_ok && llt.info() != Eigen::Success)
            *reduced_hessian_ok = false;
        Eigen::Vector<Scalar, N> p = llt.solve(-g_k);
        return {p, Eigen::Vector<Scalar, Mw>{}};
    }

    // Fully constrained: p = 0, compute multipliers only.
    // Rank-revealing ColPivHouseholderQR: a working set that reached here
    // with dependent rows would otherwise yield NaN multipliers from the
    // unpivoted solve (see initial_working_set for the failure instance).
    if(m >= n)
    {
        Eigen::Vector<Scalar, N> p = Eigen::Vector<Scalar, N>::Zero(n);
        // lambda from (A_W)^T lambda = g_k  (eq. 16.30: sum a_i lambda_i = g)
        auto qr = A_W.transpose().colPivHouseholderQr();
        Eigen::Vector<Scalar, Mw> lambda = qr.solve(g_k);
        return {p, lambda};
    }

    // QR factorization of A_W^T to get null-space basis Z.
    // A_W^T = Q * [R; 0], so Z = last (n-m) columns of Q.
    // Reference: N&W eq. 16.37.
    Eigen::HouseholderQR<Eigen::Matrix<Scalar, N, Mw>> qr(A_W.transpose());
    const int rank = m;  // working set is full rank by construction
    const int nz = n - rank;

    // Materialize only the Q columns we need via thin products.
    // Q * [0; I_{nz}] gives the null-space basis Z (n x nz).
    // Q * [I_{rank}; 0] gives the range-space basis Y (n x rank).
    Eigen::Matrix<Scalar, N, Eigen::Dynamic> Z(n, nz);
    Z.setZero();
    Z.bottomRightCorner(nz, nz).setIdentity();
    Z.applyOnTheLeft(qr.householderQ());

    Eigen::Matrix<Scalar, N, Eigen::Dynamic> Y(n, rank);
    Y.setZero();
    Y.topLeftCorner(rank, rank).setIdentity();
    Y.applyOnTheLeft(qr.householderQ());

    // Solve for p_y: A_W * Y * p_y = 0 (since constraints are A_W p = 0)
    // => p_y = 0 (the subproblem has RHS = 0 because we work with p = x - x_k)
    // So p = Z * p_z where (Z^T G Z) p_z = -Z^T g_k  (eq. 16.18)

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ZtGZ(nz, nz);
    ZtGZ.noalias() = Z.transpose() * G * Z;
    Eigen::LLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> llt(std::move(ZtGZ));

    // Check inertia: if the reduced Hessian Z^T G Z is not positive definite
    // the LLT fails and the subproblem direction is unreliable (the QP is
    // non-convex on the current null space). Surface it to the caller instead
    // of consuming a NaN solve. For the convex contract this never fires.
    if(reduced_hessian_ok && llt.info() != Eigen::Success)
        *reduced_hessian_ok = false;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs(nz);
    rhs.noalias() = -(Z.transpose() * g_k);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> p_z = llt.solve(rhs);
    Eigen::Vector<Scalar, N> p(n);
    p.noalias() = Z * p_z;

    // Multipliers: (A_W Y)^T lambda = Y^T (g_k + G p)  (eq. 16.19)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs_lam(rank);
    rhs_lam.noalias() = Y.transpose() * (g_k + G * p);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> AY(m, rank);
    AY.noalias() = A_W * Y;
    auto qr_ay = AY.transpose().colPivHouseholderQr();
    Eigen::Vector<Scalar, Mw> lambda = qr_ay.solve(rhs_lam);

    return {p, lambda};
}

// Compute blocking step length per N&W eq. 16.29.
//
// alpha_k = min(1, min_{i not in W, a_i^T p < 0} (b_i - a_i^T x) / (a_i^T p))
//
// Returns (alpha, blocking constraint index). alpha=1 and index=-1 if no blocking.
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
std::pair<Scalar, int> blocking_step_length(
    const Eigen::Vector<Scalar, N>& x_k,
    const Eigen::Vector<Scalar, N>& p_k,
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const Eigen::Vector<Scalar, M>& b_full,
    const std::vector<int>& working_set,
    int n_eq)
{
    Scalar alpha = Scalar(1);
    int blocking_idx = -1;
    const int m = static_cast<int>(A_full.rows());

    for(int i = n_eq; i < m; ++i)
    {
        // Skip constraints already in working set
        bool in_W = false;
        for(int w : working_set)
        {
            if(w == i) { in_W = true; break; }
        }
        if(in_W) continue;

        Scalar atp = A_full.row(i).dot(p_k);
        if(atp < Scalar(0))
        {
            Scalar slack = A_full.row(i).dot(x_k) - b_full[i];
            Scalar ratio = -slack / atp;
            if(ratio < alpha)
            {
                alpha = ratio;
                blocking_idx = i;
            }
        }
    }

    alpha = std::max(alpha, Scalar(0));
    return {alpha, blocking_idx};
}

// Choose the inequality constraint to leave the working set.
// Equality constraints (indices < n_eq) are never candidates for removal.
// Returns -1 (optimal for this working set) if all inequality multipliers
// are >= -tolerance.
//
// Default rule: most negative multiplier (fastest descent). When `bland` is
// set the selection switches to the eligible constraint of LOWEST global
// index (Bland's rule), which provably prevents cycling on degenerate
// vertices at the cost of slower progress.
//
// Reference: N&W Algorithm 16.1 (multiplier check step); Bland (1977) via
//            N&W Section 13.5 (anti-cycling pivot selection).
template <typename Scalar, int Mw = argmin::dynamic_dimension>
int most_negative_multiplier(
    const Eigen::Vector<Scalar, Mw>& lambda_W,
    const std::vector<int>& working_set,
    int n_eq,
    Scalar tolerance,
    bool bland = false)
{
    const int wsize = static_cast<int>(working_set.size());

    if(bland)
    {
        int drop_idx = -1;
        int best_global = std::numeric_limits<int>::max();
        for(int k = 0; k < wsize; ++k)
        {
            if(working_set[static_cast<std::size_t>(k)] < n_eq) continue;
            if(k < lambda_W.size() && lambda_W[k] < -tolerance
               && working_set[static_cast<std::size_t>(k)] < best_global)
            {
                best_global = working_set[static_cast<std::size_t>(k)];
                drop_idx = k;
            }
        }
        return drop_idx;
    }

    Scalar min_val = -tolerance;
    int drop_idx = -1;
    for(int k = 0; k < wsize; ++k)
    {
        if(working_set[static_cast<std::size_t>(k)] < n_eq) continue;
        if(k < lambda_W.size() && lambda_W[k] < min_val)
        {
            min_val = lambda_W[k];
            drop_idx = k;
        }
    }
    return drop_idx;
}

// Extract rows of A_full corresponding to working set indices.
template <typename Scalar, int M = argmin::dynamic_dimension, int N = argmin::dynamic_dimension>
Eigen::Matrix<Scalar, Eigen::Dynamic, N> extract_working_rows(
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const std::vector<int>& working_set)
{
    const int n = static_cast<int>(A_full.cols());
    const int wsize = static_cast<int>(working_set.size());
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_W(wsize, n);
    for(int k = 0; k < wsize; ++k)
        A_W.row(k) = A_full.row(working_set[static_cast<std::size_t>(k)]);
    return A_W;
}

// Build full multiplier vector from working set multipliers.
// Non-active constraint multipliers are set to zero.
template <typename Scalar, int Mw = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
Eigen::Vector<Scalar, M> build_full_lambda(
    const Eigen::Vector<Scalar, Mw>& lambda_W,
    const std::vector<int>& working_set,
    int total_constraints)
{
    Eigen::Vector<Scalar, M> lambda = Eigen::Vector<Scalar, M>::Zero(total_constraints);
    const int wsize = static_cast<int>(working_set.size());
    for(int k = 0; k < wsize && k < lambda_W.size(); ++k)
        lambda[working_set[static_cast<std::size_t>(k)]] = lambda_W[k];
    return lambda;
}

// In-place overload: writes into pre-allocated output vector.
template <typename Scalar, int Mw = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
void build_full_lambda(
    const Eigen::Vector<Scalar, Mw>& lambda_W,
    const std::vector<int>& working_set,
    int total_constraints,
    Eigen::Vector<Scalar, M>& lambda_full)
{
    lambda_full.setZero(total_constraints);
    const int wsize = static_cast<int>(working_set.size());
    for(int k = 0; k < wsize && k < lambda_W.size(); ++k)
        lambda_full[working_set[static_cast<std::size_t>(k)]] = lambda_W[k];
}

// In-place initial_working_set: writes into pre-allocated output vector.
// Applies the same rank-revealing admission as the value-returning overload
// (dependent gradient rows are skipped so A_W keeps full row rank).
template <typename Scalar, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
void initial_working_set(
    const Eigen::Vector<Scalar, N>& x0,
    const Eigen::Matrix<Scalar, M, N>& A_full,
    const Eigen::Vector<Scalar, M>& b_full,
    int n_eq,
    Scalar tolerance,
    std::vector<int>& W)
{
    W.clear();
    const int m = static_cast<int>(A_full.rows());
    const int n = static_cast<int>(A_full.cols());

    std::vector<int> candidates;
    candidates.reserve(m);
    for(int i = 0; i < n_eq; ++i)
        candidates.push_back(i);
    for(int i = n_eq; i < m; ++i)
    {
        Scalar residual = A_full.row(i).dot(x0) - b_full[i];
        if(std::abs(residual) < tolerance)
            candidates.push_back(i);
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_W(0, n);
    int rank = 0;
    for(int idx : candidates)
    {
        if(rank >= n) break;
        Eigen::Matrix<Scalar, Eigen::Dynamic, N> trial(A_W.rows() + 1, n);
        if(A_W.rows() > 0) trial.topRows(A_W.rows()) = A_W;
        trial.row(A_W.rows()) = A_full.row(idx);
        Eigen::ColPivHouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, N>> qr(trial);
        const int new_rank = static_cast<int>(qr.rank());
        if(new_rank > rank)
        {
            A_W = std::move(trial);
            rank = new_rank;
            W.push_back(idx);
        }
    }
}

// Stateful dense active-set QP solver with pre-allocated workspace.
//
// Pre-allocates factorization objects and workspace matrices at construction,
// then recomputes via compute() on each solve. Eliminates per-call dynamic
// allocation for QR, LDLT, working-set matrix extraction, and intermediate
// vectors.
//
// Template parameters:
//   N — problem dimension (compile-time or dynamic_dimension)
//   M — constraint count excluding box bounds (compile-time or dynamic_dimension)
//
// When both N and M are compile-time, the result multipliers (result_type,
// bounded by max_total_constraints = M + 2*N) and the rank-scan QR get
// inline max-bounded storage, so a warm steady-state solve performs no
// heap allocation. The arithmetic workspace stays heap-backed Dynamic on
// purpose: max-bounded compute types change Eigen's kernel selection and
// are not bit-identical to the dynamic kernels (see max_total_constraints).
//
// Reference: N&W Algorithm 16.1, pp. 460-463 (active-set method for convex QP)
//            N&W Section 16.2, eq. 16.16-16.19 (null-space method)
template <typename Scalar = double, int N = argmin::dynamic_dimension, int M = argmin::dynamic_dimension>
class active_set_qp_solver
{
    // Compile-time bound on the total constraint count (M general
    // constraints + N lower + N upper box rows). Used ONLY for the result
    // multiplier storage (result_type below), never for the compute
    // workspace: giving the workspace matrices compile-time max dimensions
    // changes Eigen's kernel selection (the product and traversal choices
    // key on MaxRows/MaxCols), which was measured to drift the solve
    // trajectory bits on mid-size problems. The result multipliers see only
    // copies and order-independent max reductions, so bounding them cannot
    // perturb the solve. Bounds past Eigen's stack-allocation limit keep
    // the heap-backed dynamic result storage.
    static constexpr bool bounds_fit_inline =
        M != argmin::dynamic_dimension && N != argmin::dynamic_dimension
        && static_cast<long long>(M + 2 * N)
                   * static_cast<long long>(sizeof(Scalar))
               <= EIGEN_STACK_ALLOCATION_LIMIT;

    static constexpr int max_total_constraints =
        bounds_fit_inline ? M + 2 * N : Eigen::Dynamic;

    // Max-bounded matrix type: rows up to MaxR, cols up to MaxC.
    template <int Rows, int Cols, int MaxR, int MaxC>
    using bounded_matrix = Eigen::Matrix<Scalar, Rows, Cols, 0, MaxR, MaxC>;

    // Max-bounded vector type: size up to MaxN.
    template <int Size, int MaxN>
    using bounded_vector = Eigen::Matrix<Scalar, Size, 1, 0, MaxN, 1>;

    // Constraint-axis workspace types deliberately keep Dynamic max
    // dimensions even when M is compile-time (see max_total_constraints):
    // these matrices feed real arithmetic, and Eigen's kernel selection on
    // max-bounded types is not bit-identical to the dynamic kernels. Their
    // storage is pre-allocated once at construction, so the dynamic shape
    // costs no steady-state allocation.
    // Constraint-row matrix: Dynamic rows, N cols.
    using constraint_matrix = bounded_matrix<Eigen::Dynamic, N, Eigen::Dynamic, N>;

    // Constraint-length vector: Dynamic size.
    using constraint_vector = bounded_vector<Eigen::Dynamic, Eigen::Dynamic>;

    // N-dimension square matrix (max-bounded).
    using n_square_matrix = bounded_matrix<Eigen::Dynamic, Eigen::Dynamic, N, N>;

    // N-dimension vector (max-bounded).
    using n_vector = bounded_vector<Eigen::Dynamic, N>;

    // QR input for A_W^T: always N rows, variable cols. Fixed-row when N is compile-time.
    using qr_matrix = bounded_matrix<N, Eigen::Dynamic, N, Eigen::Dynamic>;

    // QR input for multiplier solves: variable rows (up to N), variable cols.
    // Must be dynamic-row because (A_W * Y)^T has rank rows where rank <= N.
    using qr_lambda_matrix = bounded_matrix<Eigen::Dynamic, Eigen::Dynamic, N, Eigen::Dynamic>;

    // Rank-scan QR input: the working-set admission scan factorizes only
    // the fixed n x n padded block iws_pad_, so this type carries the N
    // max bound on both axes. With N compile-time the decomposition's
    // internal temporaries live in inline storage (no per-scan heap
    // allocation). The only consumed output is the integer rank(), so the
    // max bound cannot perturb the solve trajectory.
    using iws_qr_matrix = bounded_matrix<Eigen::Dynamic, N, N, N>;

    // Q matrix: N x N (max-bounded).
    using q_matrix = bounded_matrix<Eigen::Dynamic, Eigen::Dynamic, N, N>;

    // Periodic full refactorization interval to prevent accumulated
    // Givens rotation floating-point drift.
    static constexpr uint16_t refactorization_interval{20};

public:
    // Result type matching this solver's compile-time bounds. The lambda
    // member stays runtime-sized (box-constrained solves return
    // m_eq + m_ineq + 2n multipliers, others m_eq + m_ineq) but is
    // max-bounded by max_total_constraints, so when the bounds are
    // compile-time a per-call result local costs no heap allocation.
    using result_type =
        qp_result<Scalar, N, argmin::dynamic_dimension, max_total_constraints>;

    explicit active_set_qp_solver(int n, int max_constraints)
        : n_(n)
        , max_m_(max_constraints)
        , A_full_(max_constraints, n)
        , b_full_(max_constraints)
        , A_W_(max_constraints, n)
        , qr_lambda_(n, n)
        , qr_rr_(n, n)
        , qr_full_(n, n)
        , llt_reduced_(n)
        , Q_explicit_(n, n)
        , R_(n, max_constraints)
        , ZtGZ_(n, n)
        , ZtGZ_padded_(n, n)
        , qr_full_in_(n, n)
        , AY_pad_(n, n)
        , iws_pad_(n, n)
        , rhs_(n)
        , rhs_padded_(n)
        , residual_(n)
        , p_z_(n)
        , rhs_lam_(n)
        , g_k_(n)
        , x_(n)
        , p_sub_(n)
        , lambda_sub_(max_constraints)
        , AY_(n, n)
        , last_lambda_W_(max_constraints)
        , iws_trial_(max_constraints, n)
        , iws_qr_(n, n)
        , lambda_full_(max_constraints)
        , A_aug_(max_constraints, n)
        , b_aug_(max_constraints)
    {
        W_.reserve(static_cast<std::size_t>(max_constraints));
        last_W_.reserve(static_cast<std::size_t>(max_constraints));
        iws_candidates_.reserve(static_cast<std::size_t>(max_constraints));
    }

    active_set_qp_solver() = default;

    // Solve QP subproblem using pre-allocated workspace.
    //
    // Accepts any x0; if x0 violates A_eq * x0 = b_eq within `tol`, a
    // minimum-norm phase-1 correction projects x onto the equality
    // manifold per N&W Section 16.1's feasibility invariant before the
    // active-set loop begins. The active-set algorithm itself assumes
    // the iterate is feasible w.r.t. the working-set equalities (the
    // m >= n branch in solve_equality_subproblem returns p = 0 only
    // under that precondition).
    //
    // Preconditions:
    //   - A_eq is of shape m_eq x n with m_eq <= n; over-determined
    //     equality systems are unsupported and will assert in debug
    //     builds (the phase-1 thin-QR projection writes into a buffer
    //     of size n and uses a topLeftCorner(m_eq, m_eq) triangular
    //     view, both of which require m_eq <= n).
    //
    // Reference: N&W Section 16.1, Algorithm 16.1, pp. 460-463.
    template <int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
    result_type solve(
        const Eigen::Matrix<Scalar, N, N>& G,
        const Eigen::Vector<Scalar, N>& d,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& x0,
        const argmin::qp_options& opts = {})
    {
        const int n = static_cast<int>(G.rows());
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_ineq = static_cast<int>(A_ineq.rows());
        assemble_constraints(m_eq, m_ineq, A_eq, b_eq, A_ineq, b_ineq);
        run_active_set(n, m_eq, m_ineq, G, d, x0, opts);
        result_type res;
        write_result(res, n, m_eq + m_ineq);
        return res;
    }

    // Fill-into general overload: writes the solution into a caller-owned
    // qp_result whose x / lambda storage is reused across calls, so a warm
    // steady-state solve performs no heap allocation.
    template <int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
    void solve_into(
        const Eigen::Matrix<Scalar, N, N>& G,
        const Eigen::Vector<Scalar, N>& d,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& x0,
        const argmin::qp_options& opts,
        result_type& out)
    {
        const int n = static_cast<int>(G.rows());
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_ineq = static_cast<int>(A_ineq.rows());
        assemble_constraints(m_eq, m_ineq, A_eq, b_eq, A_ineq, b_ineq);
        run_active_set(n, m_eq, m_ineq, G, d, x0, opts);
        write_result(out, n, m_eq + m_ineq);
    }

    // Box-constraint overload. Folds box bounds into the augmented
    // inequality block and forwards to the general path, so it inherits the
    // phase-1 feasibility projection automatically.
    //
    // Reference: N&W Section 16.1, Algorithm 16.1, pp. 460-463.
    template <int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
    result_type solve(
        const Eigen::Matrix<Scalar, N, N>& G,
        const Eigen::Vector<Scalar, N>& d,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        const Eigen::Vector<Scalar, N>& x0,
        const argmin::qp_options& opts = {})
    {
        const int n = static_cast<int>(G.rows());
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_aug = assemble_box(n, A_eq, b_eq, A_ineq, b_ineq, lower, upper);
        run_active_set(n, m_eq, m_aug, G, d, x0, opts);
        result_type res;
        write_result(res, n, m_eq + m_aug);
        return res;
    }

    // Fill-into box overload (see the non-box solve_into rationale).
    template <int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
    void solve_into(
        const Eigen::Matrix<Scalar, N, N>& G,
        const Eigen::Vector<Scalar, N>& d,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper,
        const Eigen::Vector<Scalar, N>& x0,
        const argmin::qp_options& opts,
        result_type& out)
    {
        const int n = static_cast<int>(G.rows());
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_aug = assemble_box(n, A_eq, b_eq, A_ineq, b_ineq, lower, upper);
        run_active_set(n, m_eq, m_aug, G, d, x0, opts);
        write_result(out, n, m_eq + m_aug);
    }

private:
    // Assemble [A_eq; A_ineq] / [b_eq; b_ineq] into the member constraint
    // block. Equalities occupy the first m_eq rows; inequalities follow.
    template <int Meq, int Mineq>
    void assemble_constraints(
        int m_eq, int m_ineq,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq)
    {
        if(m_eq > 0)
        {
            A_full_.topRows(m_eq) = A_eq;
            b_full_.head(m_eq) = b_eq;
        }
        if(m_ineq > 0)
        {
            A_full_.middleRows(m_eq, m_ineq) = A_ineq;
            b_full_.segment(m_eq, m_ineq) = b_ineq;
        }
    }

    // Assemble the box-augmented inequality system directly into A_full_ /
    // b_full_ ([A_eq; A_ineq; I; -I] against [b_eq; b_ineq; l; -u]). Returns
    // the augmented inequality count (m_ineq + 2n). No dynamic temporaries:
    // the identity blocks are written as nullary expressions.
    template <int Meq, int Mineq>
    int assemble_box(
        int n,
        const Eigen::Matrix<Scalar, Meq, N>& A_eq,
        const Eigen::Vector<Scalar, Meq>& b_eq,
        const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
        const Eigen::Vector<Scalar, Mineq>& b_ineq,
        const Eigen::Vector<Scalar, N>& lower,
        const Eigen::Vector<Scalar, N>& upper)
    {
        const int m_eq = static_cast<int>(A_eq.rows());
        const int m_ineq = static_cast<int>(A_ineq.rows());
        const int m_aug = m_ineq + 2 * n;

        if(m_eq > 0)
        {
            A_full_.topRows(m_eq) = A_eq;
            b_full_.head(m_eq) = b_eq;
        }
        if(m_ineq > 0)
        {
            A_full_.middleRows(m_eq, m_ineq) = A_ineq;
            b_full_.segment(m_eq, m_ineq) = b_ineq;
        }
        A_full_.middleRows(m_eq + m_ineq, n) =
            Eigen::Matrix<Scalar, N, N>::Identity(n, n);
        b_full_.segment(m_eq + m_ineq, n) = lower;
        A_full_.middleRows(m_eq + m_ineq + n, n) =
            -Eigen::Matrix<Scalar, N, N>::Identity(n, n);
        b_full_.segment(m_eq + m_ineq + n, n) = -upper;
        return m_aug;
    }

    // Copy the last active-set result into a qp_result. out.x / out.lambda are
    // resized to (n) / (m_total); when the caller reuses the same result the
    // assignment reuses storage (allocation-free steady state).
    void write_result(result_type& out, int n, int m_total) const
    {
        out.x = x_.head(n);
        out.lambda = lambda_full_.head(m_total);
        out.status = status_;
        out.iterations = iters_;
        out.relaxation_factor = Scalar(0);
    }

    // Run the active-set method over the pre-assembled member constraint
    // block A_full_ / b_full_ (first m_eq rows equality, next m_ineq rows
    // inequality). Fills the iterate x_, the full multipliers lambda_full_,
    // and the run status_ / iters_. All working buffers are members, so a
    // warm steady-state solve is heap-allocation-free.
    //
    // Reference: N&W Section 16.1, Algorithm 16.1, pp. 460-463.
    void run_active_set(
        int n, int m_eq, int m_ineq,
        const Eigen::Ref<const n_square_matrix>& G,
        const Eigen::Ref<const n_vector>& d,
        const Eigen::Ref<const n_vector>& x0,
        const argmin::qp_options& opts)
    {
        const int m_total = m_eq + m_ineq;
        const int max_iter = static_cast<int>(opts.max_iterations);
        const auto tol = static_cast<Scalar>(opts.tolerance);

        compute_initial_working_set(x0, m_total, n, m_eq, tol);

        x_.head(n) = x0;
        auto x = x_.head(n);

        // Phase-1 feasibility projection. The active-set machinery in
        // solve_equality_subproblem_into assumes x is feasible w.r.t. the
        // working-set equalities (m>=n branch returns p=0 unconditionally).
        // Restore that invariant at entry by projecting any infeasible x0
        // onto the equality manifold via the minimum-norm correction
        //   dx = A_eq^T (A_eq A_eq^T)^{-1} (b_eq - A_eq x)
        // with a thin QR factorization of A_eq^T (the equality rows live in
        // A_full_.topRows(m_eq)). residual_ / rhs_ are members so the test +
        // projection allocate nothing.
        // Reference: N&W 2e Section 16.1, Algorithm 16.1, pp. 460-463
        //            (feasibility invariant); N&W 2e Section 10.3 / Lemma 16.5
        //            (minimum-norm under-determined solution).
        if(m_eq > 0)
        {
            // Precondition: A_eq must be of shape m_eq x n with m_eq <= n.
            assert(m_eq <= n
                   && "active_set_qp_solver: m_eq must be <= n for phase-1 feasibility projection");

            residual_.head(m_eq).noalias() =
                b_full_.head(m_eq) - A_full_.topRows(m_eq).leftCols(n) * x;
            if(residual_.head(m_eq).cwiseAbs().maxCoeff() > tol)
            {
                qr_lambda_.compute(A_full_.topRows(m_eq).leftCols(n).transpose());
                rhs_.head(n).setZero();
                rhs_.head(m_eq) = qr_lambda_.matrixQR()
                    .topLeftCorner(m_eq, m_eq)
                    .template triangularView<Eigen::Upper>()
                    .transpose()
                    .solve(residual_.head(m_eq));
                rhs_.head(n).applyOnTheLeft(qr_lambda_.householderQ());
                x += rhs_.head(n);
            }
        }

        qr_valid_ = false;
        givens_update_count_ = 0;

        // Anti-cycling budget: switch to Bland's lowest-index leaving rule
        // after 2*(n + m) iterations (Bland 1977; N&W Section 13.5).
        const int bland_threshold = 2 * (n + m_total);

        for(int iter = 0; iter < max_iter; ++iter)
        {
            const bool bland = iter >= bland_threshold;
            g_k_.head(n).noalias() = G * x + d;

            const int wsize = static_cast<int>(W_.size());
            for(int k = 0; k < wsize; ++k)
                A_W_.row(k) = A_full_.row(W_[static_cast<std::size_t>(k)]);

            bool reduced_hessian_ok = true;
            solve_equality_subproblem_into(
                G, g_k_.head(n), A_W_.topRows(wsize), n, wsize,
                &reduced_hessian_ok);
            auto p = p_sub_.head(n);

            // Snapshot the working-set multipliers / indices for the
            // max-iteration exit (member storage, no allocation).
            last_lambda_W_.head(wsize) = lambda_sub_.head(wsize);
            last_W_ = W_;

            if(!reduced_hessian_ok)
            {
                build_full_lambda(lambda_sub_, W_, m_total, lambda_full_);
                status_ = qp_status::indefinite_hessian;
                iters_ = iter;
                return;
            }

            if(p.norm() < tol)
            {
                int drop = most_negative_multiplier(lambda_sub_, W_, m_eq, tol, bland);
                if(drop == -1)
                {
                    build_full_lambda(lambda_sub_, W_, m_total, lambda_full_);
                    status_ = qp_status::optimal;
                    iters_ = iter;
                    return;
                }
                // Remove constraint: Givens QR downdate
                if(qr_valid_ && wsize > 1)
                {
                    remove_constraint_qr<Scalar>(
                        Q_explicit_, R_, drop, wsize);
                    ++givens_update_count_;
                }
                else
                {
                    qr_valid_ = false;
                }
                W_.erase(W_.begin() + drop);
            }
            else
            {
                auto [alpha, blocking_idx] = blocking_step_length_member(
                    x, p_sub_.head(n), m_total, n, m_eq);
                x += alpha * p;

                if(alpha < Scalar(1) && blocking_idx >= 0)
                {
                    W_.push_back(blocking_idx);
                    // Add constraint: Givens QR update
                    const int new_wsize = static_cast<int>(W_.size());
                    if(qr_valid_ && new_wsize <= n)
                    {
                        rhs_lam_.head(n).noalias() =
                            A_full_.row(blocking_idx).head(n).transpose();
                        add_constraint_qr<Scalar>(
                            Q_explicit_, R_, rhs_lam_.head(n), new_wsize);
                        ++givens_update_count_;
                    }
                    else
                    {
                        qr_valid_ = false;
                    }
                }
            }
        }

        // Max-iteration exit: return the last computed multipliers for the
        // working set they were solved against, not a zero vector.
        build_full_lambda(last_lambda_W_, last_W_, m_total, lambda_full_);
        status_ = qp_status::max_iterations;
        iters_ = max_iter;
    }

    // Full QR factorization of A_W^T using Householder, then extract
    // explicit Q and R matrices. Called on first iteration and
    // periodically to prevent accumulated Givens drift.
    void full_qr_factorize(
        const Eigen::Ref<const constraint_matrix>& A_W,
        int n,
        int m)
    {
        // Persistent Householder QR on a fixed n x n input: A_W^T (n x m) is
        // padded with (n - m) zero columns. A zero column yields an identity
        // Householder reflector, so Q and the first m columns of R are
        // bit-identical to factorizing A_W^T alone, while the constant shape
        // reuses the decomposition storage (no resize realloc).
        qr_full_in_.setZero(n, n);
        qr_full_in_.leftCols(m) = A_W.transpose();
        qr_full_.compute(qr_full_in_);
        Q_explicit_.setIdentity(n, n);
        Q_explicit_.applyOnTheLeft(qr_full_.householderQ());
        R_.topLeftCorner(n, m).setZero();
        auto R_upper = qr_full_.matrixQR();
        for(int j = 0; j < m; ++j)
            for(int i = 0; i <= j && i < n; ++i)
                R_(i, j) = R_upper(i, j);
        qr_valid_ = true;
        givens_update_count_ = 0;
    }

    // Solve equality-constrained QP subproblem via null-space method.
    //
    // Uses explicit Q/R factorization with Givens-based incremental updates
    // when the working set changes by one constraint. Falls back to full
    // HouseholderQR on the first call and periodically to prevent drift.
    //
    // Reference: N&W Section 16.2, eq. 16.16-16.19.
    //            N&W Algorithm 16.3 (QR update via Givens rotations).
    // Determine the initial working set from the (already-assembled) member
    // constraint block A_full_ / b_full_, writing indices into W_. Uses member
    // scratch (iws_trial_ / iws_qr_ / iws_candidates_) so no per-solve heap
    // allocation for the rank-revealing admission scan. Same admission rule as
    // the free initial_working_set: equalities first, then active inequalities
    // whose gradient row is independent of the retained basis.
    void compute_initial_working_set(
        const Eigen::Ref<const n_vector>& x,
        int m_total, int n, int m_eq, Scalar tol)
    {
        W_.clear();
        iws_candidates_.clear();
        for(int i = 0; i < m_eq; ++i)
            iws_candidates_.push_back(i);
        for(int i = m_eq; i < m_total; ++i)
        {
            Scalar residual = A_full_.row(i).head(n).dot(x) - b_full_[i];
            if(std::abs(residual) < tol)
                iws_candidates_.push_back(i);
        }

        int rank = 0;
        for(int idx : iws_candidates_)
        {
            if(rank >= n) break;
            iws_trial_.row(rank) = A_full_.row(idx);
            // Pad the (rank + 1) x n candidate block to a fixed n x n input
            // with zero rows (which add no rank) so the rank-revealing QR
            // factorizes a constant shape. The prescribed threshold reproduces
            // the (rank + 1)-row default (epsilon * diagonalSize), so rank()
            // returns the same integer as the unpadded decomposition and the
            // admission decision is unchanged.
            iws_pad_.setZero(n, n);
            iws_pad_.topRows(rank + 1) = iws_trial_.topRows(rank + 1);
            iws_qr_.setThreshold(
                Eigen::NumTraits<Scalar>::epsilon() * Scalar(rank + 1));
            iws_qr_.compute(iws_pad_);
            const int new_rank = static_cast<int>(iws_qr_.rank());
            if(new_rank > rank)
            {
                rank = new_rank;
                W_.push_back(idx);
            }
        }
    }

    // Blocking step length (N&W eq. 16.29) over the member constraint block.
    // Reads A_full_ / b_full_ / W_ directly so no per-iteration working-set
    // matrix copy is materialized. Returns (alpha, blocking index).
    std::pair<Scalar, int> blocking_step_length_member(
        const Eigen::Ref<const n_vector>& x_k,
        const Eigen::Ref<const n_vector>& p_k,
        int m_total, int n, int n_eq) const
    {
        Scalar alpha = Scalar(1);
        int blocking_idx = -1;
        for(int i = n_eq; i < m_total; ++i)
        {
            bool in_W = false;
            for(int w : W_)
                if(w == i) { in_W = true; break; }
            if(in_W) continue;

            Scalar atp = A_full_.row(i).head(n).dot(p_k);
            if(atp < Scalar(0))
            {
                Scalar slack = A_full_.row(i).head(n).dot(x_k) - b_full_[i];
                Scalar ratio = -slack / atp;
                if(ratio < alpha)
                {
                    alpha = ratio;
                    blocking_idx = i;
                }
            }
        }
        alpha = std::max(alpha, Scalar(0));
        return {alpha, blocking_idx};
    }

    // Fill-into equality-subproblem solve. Writes the step into p_sub_.head(n)
    // and the working-set multipliers into lambda_sub_.head(m); the caller
    // reads those persistent segments. All intermediate matrices/vectors are
    // member buffers so no per-active-set-iteration heap allocation occurs.
    void solve_equality_subproblem_into(
        const Eigen::Ref<const n_square_matrix>& G,
        const Eigen::Ref<const n_vector>& g_k,
        const Eigen::Ref<const constraint_matrix>& A_W,
        int n,
        int m,
        bool* reduced_hessian_ok = nullptr)
    {
        if(reduced_hessian_ok) *reduced_hessian_ok = true;

        if(m == 0)
        {
            llt_reduced_.compute(G);
            if(reduced_hessian_ok && llt_reduced_.info() != Eigen::Success)
                *reduced_hessian_ok = false;
            p_sub_.head(n) = llt_reduced_.solve(-g_k);
            qr_valid_ = false;
            return;
        }

        if(m >= n)
        {
            // Rank-revealing solve: a dependent working-set row would make the
            // unpivoted QR produce NaN multipliers. ColPivHouseholderQR keeps
            // the least-squares multipliers finite on a degenerate vertex.
            p_sub_.head(n).setZero();
            qr_rr_.compute(A_W.transpose());
            lambda_sub_.head(m) = qr_rr_.solve(g_k);
            return;
        }

        // Full refactorization if not valid or periodic drift prevention
        if(!qr_valid_ || givens_update_count_ >= refactorization_interval)
            full_qr_factorize(A_W, n, m);

        const int rank = m;
        const int nz = n - rank;
        auto Y_block = Q_explicit_.leftCols(rank);
        auto Z_block = Q_explicit_.rightCols(nz);

        ZtGZ_.topLeftCorner(nz, nz).noalias() =
            Z_block.transpose() * G * Z_block;
        // Factor a fixed n x n block-diagonal system [Z^T G Z, 0; 0, I] so the
        // reduced-Hessian LLT always computes the same shape (no resize
        // realloc). The identity trailing block does not couple into the
        // top-left solve, so the reduced-space direction is bit-identical to
        // factoring Z^T G Z alone.
        ZtGZ_padded_.setZero(n, n);
        ZtGZ_padded_.topLeftCorner(nz, nz) = ZtGZ_.topLeftCorner(nz, nz);
        if(nz < n)
            ZtGZ_padded_.bottomRightCorner(n - nz, n - nz).setIdentity();
        llt_reduced_.compute(ZtGZ_padded_);
        if(reduced_hessian_ok && llt_reduced_.info() != Eigen::Success)
            *reduced_hessian_ok = false;

        rhs_.head(nz).noalias() = -(Z_block.transpose() * g_k);
        rhs_padded_.head(n).setZero();
        rhs_padded_.head(nz) = rhs_.head(nz);
        p_z_.head(n) = llt_reduced_.solve(rhs_padded_.head(n));
        p_sub_.head(n).noalias() = Z_block * p_z_.head(nz);

        AY_.topLeftCorner(m, rank).noalias() = A_W * Y_block;
        rhs_lam_.head(rank).noalias() =
            Y_block.transpose() * (g_k + G * p_sub_.head(n));
        // Pad AY^T (rank x rank, since rank == m) to a fixed n x n block-
        // diagonal system [AY^T, 0; 0, I] and solve [rhs_lam; 0]. Column
        // pivoting decouples across the zero off-diagonal blocks, so the
        // top-block multiplier solution is bit-identical to solving AY^T alone
        // while the constant n x n shape reuses the ColPiv storage.
        AY_pad_.setZero(n, n);
        AY_pad_.topLeftCorner(rank, m) = AY_.topLeftCorner(m, rank).transpose();
        if(rank < n)
            AY_pad_.bottomRightCorner(n - rank, n - rank).setIdentity();
        rhs_padded_.head(n).setZero();
        rhs_padded_.head(rank) = rhs_lam_.head(rank);
        qr_rr_.compute(AY_pad_);
        p_z_.head(n) = qr_rr_.solve(rhs_padded_.head(n));
        lambda_sub_.head(rank) = p_z_.head(rank);
    }

    int n_{0};
    int max_m_{0};

    constraint_matrix A_full_;
    constraint_vector b_full_;
    constraint_matrix A_W_;

    Eigen::HouseholderQR<qr_lambda_matrix> qr_lambda_;
    Eigen::ColPivHouseholderQR<qr_lambda_matrix> qr_rr_;
    Eigen::HouseholderQR<qr_matrix> qr_full_;
    Eigen::LLT<n_square_matrix> llt_reduced_;
    q_matrix Q_explicit_;
    qr_matrix R_;
    bool qr_valid_{false};
    uint16_t givens_update_count_{0};
    n_square_matrix ZtGZ_;
    n_square_matrix ZtGZ_padded_;
    // Fixed n x n padded factorization inputs so the Householder / ColPiv
    // decompositions always compute the same shape (no resize realloc). The
    // padding is structured (zero columns / rows, identity trailing block) so
    // the factorization values consumed are bit-identical to the unpadded
    // problem.
    n_square_matrix qr_full_in_;
    n_square_matrix AY_pad_;
    n_square_matrix iws_pad_;
    n_vector rhs_;
    n_vector rhs_padded_;
    n_vector residual_;
    n_vector p_z_;
    n_vector rhs_lam_;
    n_vector g_k_;
    // Result of the last active-set run (filled by run_active_set); the
    // solution iterate lives in x_ and the full multipliers in lambda_full_.
    qp_status status_{qp_status::optimal};
    int iters_{0};
    // Persistent iterate + equality-subproblem result buffers so the
    // active-set loop's per-iteration step / multiplier vectors and the
    // reduced-space AY product reuse storage instead of heap-allocating.
    n_vector x_;
    n_vector p_sub_;
    constraint_vector lambda_sub_;
    q_matrix AY_;

    std::vector<int> W_;
    // Snapshot of the multipliers / working set solved against, for the
    // max-iteration exit; members so the snapshot copy reuses storage.
    constraint_vector last_lambda_W_;
    std::vector<int> last_W_;
    // Working-set-scan scratch (rank-revealing admission), member so the
    // per-solve initial working set costs no heap allocation.
    std::vector<int> iws_candidates_;
    constraint_matrix iws_trial_;
    Eigen::ColPivHouseholderQR<iws_qr_matrix> iws_qr_;
    constraint_vector lambda_full_;

    constraint_matrix A_aug_;
    constraint_vector b_aug_;
};

// Dense active-set QP solver.
//
// Solves: min 0.5 x^T G x + d^T x
//         s.t. A_eq x = b_eq             (equality)
//              A_ineq x >= b_ineq        (inequality, lower-bound form)
//
// Convention: inequality constraints are A_ineq * x >= b_ineq.
// The initial point x0 must be feasible.
//
// Reference: N&W Algorithm 16.1, pp. 460-463.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
qp_result<Scalar, N> solve_qp(
    const Eigen::Matrix<Scalar, N, N>& G,
    const Eigen::Vector<Scalar, N>& d,
    const Eigen::Matrix<Scalar, Meq, N>& A_eq,
    const Eigen::Vector<Scalar, Meq>& b_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
    const Eigen::Vector<Scalar, Mineq>& b_ineq,
    const Eigen::Vector<Scalar, N>& x0,
    const argmin::qp_options& opts = {})
{
    const int n = static_cast<int>(G.rows());
    const int m_eq = static_cast<int>(A_eq.rows());
    const int m_ineq = static_cast<int>(A_ineq.rows());
    const int m_total = m_eq + m_ineq;
    const int max_iter = static_cast<int>(opts.max_iterations);
    const auto tol = static_cast<Scalar>(opts.tolerance);

    // Assemble full constraint matrix: [A_eq; A_ineq] x >= [b_eq; b_ineq]
    // (Equality constraints treated as a_i^T x = b_i, always active.)
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_full(m_total, n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_full(m_total);
    if(m_eq > 0)
    {
        A_full.topRows(m_eq) = A_eq;
        b_full.head(m_eq) = b_eq;
    }
    if(m_ineq > 0)
    {
        A_full.bottomRows(m_ineq) = A_ineq;
        b_full.tail(m_ineq) = b_ineq;
    }

    auto W = initial_working_set<Scalar, N>(x0, A_full, b_full, m_eq, tol);
    Eigen::Vector<Scalar, N> x = x0;

    // Anti-cycling: after this many iterations without convergence, switch
    // the leaving-constraint choice to Bland's lowest-index rule, which
    // provably terminates on degenerate vertices. 2*(n + m) is the standard
    // heuristic budget before the guard engages (Bland 1977; N&W Section 13.5).
    const int bland_threshold = 2 * (n + m_total);

    // Snapshot of the multipliers computed for the working set they were
    // solved against, so the max-iteration exit can return the last genuinely
    // computed duals instead of a zero vector.
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> last_lambda_W;
    std::vector<int> last_W;

    for(int iter = 0; iter < max_iter; ++iter)
    {
        const bool bland = iter >= bland_threshold;
        Eigen::Vector<Scalar, N> g_k = (G * x + d).eval();
        auto A_W = extract_working_rows(A_full, W);

        bool reduced_hessian_ok = true;
        auto [p, lambda_W] = solve_equality_qp<Scalar, N>(
            G, g_k, A_W, &reduced_hessian_ok);

        last_lambda_W = lambda_W;
        last_W = W;

        if(!reduced_hessian_ok)
        {
            auto lambda_full = build_full_lambda(lambda_W, W, m_total);
            qp_result<Scalar, N> res;
            res.x = x;
            res.lambda = lambda_full;
            res.status = qp_status::indefinite_hessian;
            res.iterations = iter;
            return res;
        }

        if(p.norm() < tol)
        {
            // At stationary point for current working set -- check multipliers
            int drop = most_negative_multiplier(lambda_W, W, m_eq, tol, bland);
            if(drop == -1)
            {
                // All inequality multipliers non-negative: optimal
                auto lambda_full = build_full_lambda(lambda_W, W, m_total);
                qp_result<Scalar, N> res;
                res.x = x;
                res.lambda = lambda_full;
                res.status = qp_status::optimal;
                res.iterations = iter;
                return res;
            }
            // Remove the constraint with the selected multiplier
            W.erase(W.begin() + drop);
        }
        else
        {
            // Step is nonzero: compute blocking step length (eq. 16.29)
            auto [alpha, blocking_idx] = blocking_step_length<Scalar, N>(
                x, p, A_full, b_full, W, m_eq);
            x += alpha * p;

            if(alpha < Scalar(1) && blocking_idx >= 0)
                W.push_back(blocking_idx);
        }
    }

    // Max-iteration exit: return the multipliers last computed for the
    // working set they were solved against (N&W eq. 16.30), not zeros.
    auto lambda_full = build_full_lambda(last_lambda_W, last_W, m_total);
    qp_result<Scalar, N> res;
    res.x = x;
    res.lambda = lambda_full;
    res.status = qp_status::max_iterations;
    res.iterations = max_iter;
    return res;
}

// Convenience overload with box constraints.
//
// Converts box bounds l <= x <= u to inequality constraints:
//   I*x >= l    (lower bounds)
//  -I*x >= -u   (upper bounds)
// and appends them to A_ineq/b_ineq.
template <typename Scalar = double, int N = argmin::dynamic_dimension,
          int Meq = argmin::dynamic_dimension, int Mineq = argmin::dynamic_dimension>
qp_result<Scalar, N> solve_qp(
    const Eigen::Matrix<Scalar, N, N>& G,
    const Eigen::Vector<Scalar, N>& d,
    const Eigen::Matrix<Scalar, Meq, N>& A_eq,
    const Eigen::Vector<Scalar, Meq>& b_eq,
    const Eigen::Matrix<Scalar, Mineq, N>& A_ineq,
    const Eigen::Vector<Scalar, Mineq>& b_ineq,
    const Eigen::Vector<Scalar, N>& lower,
    const Eigen::Vector<Scalar, N>& upper,
    const Eigen::Vector<Scalar, N>& x0,
    const argmin::qp_options& opts = {})
{
    const int n = static_cast<int>(G.rows());
    const int m_ineq = static_cast<int>(A_ineq.rows());
    const int m_box = 2 * n;

    // Build augmented inequality system: [A_ineq; I; -I] x >= [b_ineq; l; -u]
    Eigen::Matrix<Scalar, Eigen::Dynamic, N> A_aug(m_ineq + m_box, n);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_aug(m_ineq + m_box);

    if(m_ineq > 0)
    {
        A_aug.topRows(m_ineq) = A_ineq;
        b_aug.head(m_ineq) = b_ineq;
    }

    A_aug.middleRows(m_ineq, n) = Eigen::Matrix<Scalar, N, N>::Identity(n, n);
    b_aug.segment(m_ineq, n) = lower;

    A_aug.bottomRows(n) = -Eigen::Matrix<Scalar, N, N>::Identity(n, n);
    b_aug.tail(n) = -upper;

    return solve_qp(G, d, A_eq, b_eq, A_aug, b_aug, x0, opts);
}

}

#endif
