#ifndef HPP_GUARD_ARGMIN_QP_DETAIL_SPARSE_KKT_H
#define HPP_GUARD_ARGMIN_QP_DETAIL_SPARSE_KKT_H

// Assembly and factorization of the full quasi-definite KKT matrix
//
//   K = [[ P + sigma*I,      A^T        ],
//        [ A,           -diag(rho)^-1   ]]
//
// used by the sparse operator-splitting QP solver. Only the lower triangle is
// stored; the leading block is positive definite and the trailing block is
// negative definite, so K is quasi-definite and therefore indefinite.
//
// Eigen documents its simplicial LDL^T decomposition for symmetric positive
// definite input, whereas this matrix is indefinite: the reliance is deliberate
// and rests on quasi-definite matrices being strongly factorizable under any
// symmetric permutation (Vanderbei 1995), together with the fact that the
// simplicial decomposition performs no pivoting and therefore computes an
// LDL^T with sign-indefinite D -- the same no-pivot factorization the reference
// implementation of the algorithm uses. Because that is off Eigen's documented
// contract it is verified by an in-tree numeric witness rather than assumed,
// and the single factorization_type alias below is the one site at which the
// decomposition would be swapped for a general sparse LU should the witness
// ever fail.
//
// Reference: Vanderbei (1995), "Symmetric quasi-definite matrices," SIAM J.
//            Optim. 5(1):100-113; Stellato, Banjac, Goulart, Bemporad, Boyd
//            (2020), "OSQP: An operator splitting solver for quadratic
//            programs," Math. Prog. Comp. 12:637-672, Section 5.

#include "argmin/types.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#include <cstddef>
#include <vector>

namespace argmin::detail
{

template <typename Scalar>
class sparse_kkt
{
public:
    using sparse_type = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
    using index_type = typename sparse_type::StorageIndex;
    using factorization_type =
        Eigen::SimplicialLDLT<sparse_type, Eigen::Lower, Eigen::AMDOrdering<index_type>>;

    // Re-pose contract: analyze() may be called again on an already-analyzed
    // instance with a DIFFERENT sparsity pattern, and it rebuilds everything --
    // the assembly, the diagonal-slot table and the symbolic analysis. A slot
    // table surviving a re-pose would silently corrupt the value-array writes
    // in refactorize(), so no prior state may persist. This is the only legal
    // route by which the pattern may change.
    bool analyze(const sparse_type& P, const sparse_type& A, Scalar sigma,
                 const std::vector<Scalar>& rho_inv)
    {
        const int n = static_cast<int>(P.rows());
        const int m = static_cast<int>(A.rows());

        analyzed_ = false;
        factorized_ = false;
        n_ = 0;
        m_ = 0;
        diag_slot_.clear();
        K_ = sparse_type(n + m, n + m);

        if(static_cast<int>(P.cols()) != n || static_cast<int>(A.cols()) != n)
            return false;
        if(rho_inv.size() != static_cast<std::size_t>(m))
            return false;

        std::vector<Eigen::Triplet<Scalar, index_type>> triplets;
        triplets.reserve(static_cast<std::size_t>(P.nonZeros() + A.nonZeros() + n + m));

        // P arrives full symmetric, so only its lower triangle is taken.
        for(int j = 0; j < n; ++j)
            for(typename sparse_type::InnerIterator it(P, j); it; ++it)
                if(it.row() >= j)
                    triplets.emplace_back(static_cast<index_type>(it.row()),
                                          static_cast<index_type>(j), it.value());

        // Unconditional diagonal for every leading row: sigma always owns a
        // slot and the pattern stays independent of P's diagonal fill.
        for(int i = 0; i < n; ++i)
            triplets.emplace_back(static_cast<index_type>(i), static_cast<index_type>(i), sigma);

        for(int j = 0; j < n; ++j)
            for(typename sparse_type::InnerIterator it(A, j); it; ++it)
                triplets.emplace_back(static_cast<index_type>(n + it.row()),
                                      static_cast<index_type>(j), it.value());

        for(int i = 0; i < m; ++i)
            triplets.emplace_back(static_cast<index_type>(n + i), static_cast<index_type>(n + i),
                                  -rho_inv[static_cast<std::size_t>(i)]);

        K_.setFromTriplets(triplets.begin(), triplets.end());
        K_.makeCompressed();

        diag_slot_.assign(static_cast<std::size_t>(m), -1);
        const index_type* outer = K_.outerIndexPtr();
        const index_type* inner = K_.innerIndexPtr();
        for(int i = 0; i < m; ++i)
        {
            const index_type col = static_cast<index_type>(n + i);
            for(index_type k = outer[col]; k < outer[col + 1]; ++k)
                if(inner[k] == col)
                {
                    diag_slot_[static_cast<std::size_t>(i)] = static_cast<int>(k);
                    break;
                }
            if(diag_slot_[static_cast<std::size_t>(i)] < 0)
                return false;
        }

        ++symbolic_analyses_;
        ldlt_.analyzePattern(K_);
        if(ldlt_.info() != Eigen::Success)
            return false;

        n_ = n;
        m_ = m;

        // Size the resolve-path scratch once, here, where n_ + m_ is first
        // known. Both buffers live for the analysis and are never resized on the
        // resolve path, so solve_into allocates nothing: work_ carries the
        // permuted intermediate through the five solve steps, and dinv_ caches
        // the reciprocal of the factor diagonal D (value-refreshed in
        // refactorize(), whenever the numeric factorization changes).
        work_.resize(n + m);
        dinv_.resize(n + m);

        analyzed_ = true;
        return true;
    }

    // K_ is compressed and its pattern is fixed for the lifetime of the
    // analysis, so the value-array indices captured in analyze() stay valid and
    // a rho update is a values-only write -- no symbolic work is redone.
    bool refactorize(const std::vector<Scalar>& rho_inv)
    {
        factorized_ = false;
        if(!analyzed_ || rho_inv.size() != static_cast<std::size_t>(m_))
            return false;

        Scalar* values = K_.valuePtr();
        for(std::size_t i = 0; i < rho_inv.size(); ++i)
            values[diag_slot_[i]] = -rho_inv[i];

        ++numeric_factorizations_;
        ldlt_.factorize(K_);
        factorized_ = ldlt_.info() == Eigen::Success;
        if(factorized_)
        {
            // A rho update recomputes the factor diagonal D, so the cached
            // reciprocal must be refreshed here or a subsequent solve_into would
            // silently scale by the pre-update diagonal. vectorD() returns D by
            // value (one Eigen copy); refactorize() is off the resolve claim and
            // already allocates inside Eigen's factorize, so materializing D once
            // to invert it is acceptable.
            dinv_ = ldlt_.vectorD().cwiseInverse();
        }
        return factorized_;
    }

    // Manual permuted LDL^T solve into pre-sized member scratch, faithful to
    // Eigen's internal SimplicialCholeskyBase::_solve_impl and differing only in
    // that every step targets a reused buffer instead of an allocating temporary:
    //   work = P * rhs; L y = work; work *= 1/D; L^T z = work; out = Pinv * work.
    // The final un-permute writes into out (a distinct buffer), which keeps it on
    // Eigen's non-aliased scatter branch; an in-place aliased permutation would
    // heap-allocate a mask every call. The reciprocal diagonal is read from the
    // dinv_ cache rather than recomputed, because vectorD() would copy D by value
    // on each call. Given a warm factorization this allocates nothing.
    void solve_into(const argmin::vector<Scalar>& rhs, argmin::vector<Scalar>& out) const
    {
        // Required: sizing out is a no-op on the warm resolve path (out already
        // has rows() entries) but is the only place the polish path's reduced
        // solution vectors are sized.
        out.resize(rows());
        if(ldlt_.info() != Eigen::Success)
            return;

        // Guard the triangular solves for a degenerate fully-diagonal factor
        // (L == I, no stored off-diagonal), mirroring _solve_impl's own guard.
        const bool has_off_diagonal = ldlt_.matrixL().nestedExpression().nonZeros() > 0;

        work_ = ldlt_.permutationP() * rhs;
        if(has_off_diagonal)
            ldlt_.matrixL().solveInPlace(work_);
        work_.array() *= dinv_.array();
        if(has_off_diagonal)
            ldlt_.matrixU().solveInPlace(work_);
        out = ldlt_.permutationPinv() * work_;
    }

    const sparse_type& matrix() const { return K_; }

    // Exposes the signs the factorization actually computed, which is the only
    // way to assert that the indefinite path was exercised.
    auto diagonal_factor() const { return ldlt_.vectorD(); }

    // Monotone across re-poses and never reset: the factor-once contract of the
    // owning solver is a statement about work performed over a lifetime, and a
    // counter that a re-pose zeroed could not express it.
    std::size_t symbolic_analyses() const { return symbolic_analyses_; }
    std::size_t numeric_factorizations() const { return numeric_factorizations_; }

    int rows() const { return n_ + m_; }
    int primal_size() const { return n_; }
    int dual_size() const { return m_; }
    bool analyzed() const { return analyzed_; }
    bool factorized() const { return factorized_; }

private:
    sparse_type K_;
    factorization_type ldlt_;
    std::vector<int> diag_slot_;
    // Resolve-path scratch, sized in analyze() and (dinv_) value-refreshed in
    // refactorize(); mutable so solve_into stays const for its const callers.
    mutable argmin::vector<Scalar> work_;
    mutable argmin::vector<Scalar> dinv_;
    std::size_t symbolic_analyses_{0};
    std::size_t numeric_factorizations_{0};
    int n_{0};
    int m_{0};
    bool analyzed_{false};
    bool factorized_{false};
};

}

#endif
