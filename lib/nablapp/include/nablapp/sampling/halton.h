#ifndef HPP_GUARD_NABLAPP_SAMPLING_HALTON_H
#define HPP_GUARD_NABLAPP_SAMPLING_HALTON_H

#include <Eigen/Core>

#include <cassert>

namespace nablapp
{

// Halton quasi-random sequence generator.
//
// Produces low-discrepancy points in [0,1]^d using van der Corput
// base-reversal with the first 10 primes as bases. Useful for
// space-filling initialization of population-based optimizers
// (e.g. CMA-ES) and surrogate model sampling plans.
//
// Reference: K&W Algorithm 16.11, Section 16.7.2
//            (van der Corput base-reversal, Halton sequence).

template <typename Scalar = double>
class halton_sequence
{
public:
    explicit halton_sequence(int dim, int skip = 0)
        : dimension_{dim}
        , index_{1 + skip}
    {
        assert(dim >= 1 && dim <= 10);
    }

    Eigen::VectorX<Scalar> next()
    {
        Eigen::VectorX<Scalar> point(dimension_);
        for(int i = 0; i < dimension_; ++i)
        {
            point(i) = van_der_corput(index_, primes[i]);
        }
        ++index_;
        return point;
    }

    Eigen::MatrixX<Scalar> sample(int n)
    {
        Eigen::MatrixX<Scalar> mat(dimension_, n);
        for(int j = 0; j < n; ++j)
        {
            mat.col(j) = next();
        }
        return mat;
    }

    void reset(int skip = 0) { index_ = 1 + skip; }

    int dimension() const { return dimension_; }

    int index() const { return index_; }

private:
    static constexpr int primes[10] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

    static Scalar van_der_corput(int n, int base)
    {
        Scalar result = 0;
        Scalar f = 1;
        while(n > 0)
        {
            f /= static_cast<Scalar>(base);
            result += f * static_cast<Scalar>(n % base);
            n /= base;
        }
        return result;
    }

    int dimension_;
    int index_;
};

}

#endif
