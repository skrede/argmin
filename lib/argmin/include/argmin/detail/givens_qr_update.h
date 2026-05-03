#ifndef HPP_GUARD_ARGMIN_DETAIL_GIVENS_QR_UPDATE_H
#define HPP_GUARD_ARGMIN_DETAIL_GIVENS_QR_UPDATE_H

// Givens rotation for incremental QR factorization updates.
//
// When the active-set working set changes by one constraint, the QR
// factorization of A_W^T can be updated in O(n*m) via Givens rotations
// instead of a full O(n^2*m) recompute.
//
// Reference: Golub & Van Loan, "Matrix Computations" 4th ed., Algorithm 5.1.3
//            (Givens rotation computation).
//            N&W Algorithm 16.3 (QR update in active-set context).

#include <Eigen/Core>

#include <cmath>

namespace argmin::detail
{

// Givens rotation zeroing element b in the pair [a; b] -> [r; 0].
//
// Computes c, s such that [c s; -s c] [a; b] = [r; 0].
// Uses the numerically stable formulation from Golub & Van Loan
// Algorithm 5.1.3 to avoid overflow/underflow.
template <typename Scalar>
struct givens_rotation
{
    Scalar c;
    Scalar s;

    // Compute c, s such that [c s; -s c] [a; b] = [r; 0].
    static givens_rotation compute(Scalar a, Scalar b)
    {
        using std::abs;
        using std::sqrt;

        if(b == Scalar(0))
            return {Scalar(1), Scalar(0)};

        if(abs(b) > abs(a))
        {
            Scalar tau = a / b;
            Scalar s_val = Scalar(1) / sqrt(Scalar(1) + tau * tau);
            return {s_val * tau, s_val};
        }

        Scalar tau = b / a;
        Scalar c_val = Scalar(1) / sqrt(Scalar(1) + tau * tau);
        return {c_val, c_val * tau};
    }

    // Apply rotation to rows i, j of matrix M (left-multiply by G^T).
    // Transforms rows i and j: [row_i; row_j] <- [c s; -s c] [row_i; row_j].
    template <typename Derived>
    void apply_left(Eigen::MatrixBase<Derived>& M, int i, int j) const
    {
        for(int col = 0; col < M.cols(); ++col)
        {
            Scalar t1 = M(i, col);
            Scalar t2 = M(j, col);
            M(i, col) =  c * t1 + s * t2;
            M(j, col) = -s * t1 + c * t2;
        }
    }

    // Apply rotation to columns i, j of matrix M (right-multiply by G).
    // Transforms cols i and j: [col_i col_j] <- [col_i col_j] [c -s; s c].
    template <typename Derived>
    void apply_right(Eigen::MatrixBase<Derived>& M, int i, int j) const
    {
        for(int row = 0; row < M.rows(); ++row)
        {
            Scalar t1 = M(row, i);
            Scalar t2 = M(row, j);
            M(row, i) = c * t1 + s * t2;
            M(row, j) = -s * t1 + c * t2;
        }
    }
};

// Add a constraint row to the QR factorization of A_W^T.
//
// Given Q^T A_W^T = [R; 0] (Q is n x n, R is rank x n_cols),
// appending a new column a_new^T to A_W^T produces a new column
// in Q^T * [A_W^T | a_new] = [R | q; 0 | *] where q = Q^T a_new.
//
// Givens rotations zero out the sub-diagonal entries of the new column,
// restoring upper triangular form.
//
// Q_explicit is updated by applying the transposed rotations to its columns.
// R is updated by applying the rotations to its rows.
//
// new_rank = old_rank + 1 (the rank after adding).
//
// Reference: N&W Algorithm 16.3 (constraint addition).
template <typename Scalar, typename QMatrix, typename RMatrix>
void add_constraint_qr(
    QMatrix& Q_explicit,
    RMatrix& R,
    const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& a_new,
    int new_rank)
{
    const int n = static_cast<int>(Q_explicit.rows());
    const int col = new_rank - 1;  // column index in R for the new constraint

    // Compute new R column: R(:, col) = Q^T * a_new
    // Since Q is orthogonal and stored explicitly: new_col = Q^T * a_new
    for(int i = 0; i < n; ++i)
        R(i, col) = Q_explicit.col(i).dot(a_new);

    // Zero out entries R(n-1, col) down to R(col+1, col) using Givens rotations.
    // Each rotation zeros one sub-diagonal entry.
    for(int i = n - 1; i > col; --i)
    {
        auto g = givens_rotation<Scalar>::compute(R(i - 1, col), R(i, col));

        // Apply to R rows i-1, i (affects all columns 0..new_rank-1)
        g.apply_left(R, i - 1, i);

        // Apply transposed rotation to Q columns i-1, i
        // Q_new = Q_old * G  (right-multiply), where G is the Givens matrix
        g.apply_right(Q_explicit, i - 1, i);
    }
}

// Remove a constraint from the QR factorization of A_W^T.
//
// Removes column k from R (0-indexed among the rank columns), shifts
// columns k+1..old_rank-1 left by one, then applies Givens rotations
// to restore upper triangular form.
//
// After the shift, the matrix has upper Hessenberg form: each column j
// (for j >= k) has one sub-diagonal entry at row j+1. A single Givens
// rotation per column restores triangularity.
//
// old_rank = rank before removal. After removal, rank = old_rank - 1.
//
// Reference: N&W Algorithm 16.3 (constraint removal).
template <typename Scalar, typename QMatrix, typename RMatrix>
void remove_constraint_qr(
    QMatrix& Q_explicit,
    RMatrix& R,
    int k,
    int old_rank)
{
    // Shift R columns: move columns k+1..old_rank-1 to k..old_rank-2
    for(int j = k; j < old_rank - 1; ++j)
        R.col(j) = R.col(j + 1);

    const int new_rank = old_rank - 1;
    const int n = static_cast<int>(Q_explicit.rows());

    // After column deletion and shift, columns k..new_rank-1 have
    // one sub-diagonal entry at row j+1. Zero it with one Givens
    // rotation per column, chasing the bulge rightward.
    for(int j = k; j < new_rank && j + 1 < n; ++j)
    {
        if(R(j + 1, j) == Scalar(0))
            continue;

        auto g = givens_rotation<Scalar>::compute(R(j, j), R(j + 1, j));

        g.apply_left(R, j, j + 1);
        g.apply_right(Q_explicit, j, j + 1);
    }
}

}

#endif
