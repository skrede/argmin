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

        ldlt_.analyzePattern(K_);
        if(ldlt_.info() != Eigen::Success)
            return false;

        n_ = n;
        m_ = m;
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

        ldlt_.factorize(K_);
        factorized_ = ldlt_.info() == Eigen::Success;
        return factorized_;
    }

    void solve_into(const argmin::vector<Scalar>& rhs, argmin::vector<Scalar>& out) const
    {
        out = ldlt_.solve(rhs);
    }

    const sparse_type& matrix() const { return K_; }

    // Exposes the signs the factorization actually computed, which is the only
    // way to assert that the indefinite path was exercised.
    auto diagonal_factor() const { return ldlt_.vectorD(); }

    int rows() const { return n_ + m_; }
    int primal_size() const { return n_; }
    int dual_size() const { return m_; }
    bool analyzed() const { return analyzed_; }
    bool factorized() const { return factorized_; }

private:
    sparse_type K_;
    factorization_type ldlt_;
    std::vector<int> diag_slot_;
    int n_{0};
    int m_{0};
    bool analyzed_{false};
    bool factorized_{false};
};

}

#endif
