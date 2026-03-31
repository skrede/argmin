#ifndef HPP_GUARD_NABLAPP_HESSIAN_LBFGS_HISTORY_H
#define HPP_GUARD_NABLAPP_HESSIAN_LBFGS_HISTORY_H

#include <Eigen/Core>

#include <vector>

namespace nablapp
{

// Limited-memory BFGS (L-BFGS) history with two-loop recursion.
//
// Stores the most recent m (s, y) pairs in a circular buffer and
// computes H_k * g via the two-loop recursion without forming H_k
// explicitly. The initial Hessian scaling uses gamma from the most
// recent pair (N&W eq. 9.6).
//
// Reference: N&W Algorithm 9.1 (L-BFGS two-loop recursion),
//            N&W eq. 9.6 (initial Hessian scaling).

template <typename Scalar = double>
class lbfgs_history
{
    struct pair
    {
        Eigen::VectorX<Scalar> s;
        Eigen::VectorX<Scalar> y;
        Scalar rho;
    };

public:
    explicit lbfgs_history(int m = 10) : capacity_(m) {}

    // Store a new (s, y) pair. Pairs with s^T y <= 0 are rejected
    // (curvature condition not satisfied).
    void push(const Eigen::VectorX<Scalar>& s, const Eigen::VectorX<Scalar>& y)
    {
        Scalar sTy = s.dot(y);
        if(sTy <= Scalar(0)) return;

        // Lazy allocation on first valid push
        if(buffer_.empty())
        {
            buffer_.resize(capacity_);
        }

        int idx = (head_ + count_) % capacity_;
        if(count_ == capacity_)
        {
            // Buffer full -- overwrite oldest
            idx = head_;
            head_ = (head_ + 1) % capacity_;
        }
        else
        {
            ++count_;
        }

        buffer_[idx].s = s;
        buffer_[idx].y = y;
        buffer_[idx].rho = Scalar(1) / sTy;
    }

    // L-BFGS two-loop recursion (N&W Algorithm 9.1).
    //
    // Returns H_k * g (without negation). The caller applies the
    // negation for the search direction: d = -two_loop_recursion(g).
    //
    // If history is empty, returns g (steepest descent, caller negates).
    Eigen::VectorX<Scalar> two_loop_recursion(const Eigen::VectorX<Scalar>& g) const
    {
        if(count_ == 0) return g;

        const int k = count_;
        Eigen::VectorX<Scalar> q = g;
        std::vector<Scalar> alpha(k);

        // Backward loop: i from newest to oldest (N&W Algorithm 9.1, step 1)
        for(int j = k - 1; j >= 0; --j)
        {
            int idx = (head_ + j) % capacity_;
            alpha[j] = buffer_[idx].rho * buffer_[idx].s.dot(q);
            q -= alpha[j] * buffer_[idx].y;
        }

        // Initial Hessian scaling (N&W eq. 9.6):
        // gamma_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
        int newest = (head_ + k - 1) % capacity_;
        Scalar gamma = buffer_[newest].s.dot(buffer_[newest].y)
                     / buffer_[newest].y.squaredNorm();
        Eigen::VectorX<Scalar> r = gamma * q;

        // Forward loop: i from oldest to newest (N&W Algorithm 9.1, step 3)
        for(int j = 0; j < k; ++j)
        {
            int idx = (head_ + j) % capacity_;
            Scalar beta = buffer_[idx].rho * buffer_[idx].y.dot(r);
            r += buffer_[idx].s * (alpha[j] - beta);
        }

        return r;
    }

    void reset()
    {
        head_ = 0;
        count_ = 0;
        buffer_.clear();
    }

    int size() const { return count_; }
    int capacity() const { return capacity_; }

private:
    std::vector<pair> buffer_;
    int head_{0};
    int count_{0};
    int capacity_;
};

}

#endif
