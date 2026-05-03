#ifndef HPP_GUARD_ARGMIN_TYPES_H
#define HPP_GUARD_ARGMIN_TYPES_H

#include <Eigen/Core>

namespace argmin
{

// Vocabulary types for argmin.
//
// All concept interfaces and solver code use these aliases. Eigen expression
// templates are lazy -- do NOT use `auto` to capture Eigen expressions as
// temporaries, because the underlying storage may be invalidated before
// evaluation. Always assign to a concrete type (vector<>, matrix<>, etc.).

inline constexpr int dynamic_dimension = Eigen::Dynamic;

template <typename Scalar = double, int N = dynamic_dimension>
using vector = Eigen::Vector<Scalar, N>;

template <typename Scalar = double, int Rows = dynamic_dimension, int Cols = dynamic_dimension>
using matrix = Eigen::Matrix<Scalar, Rows, Cols>;

template <typename Scalar = double, int N = dynamic_dimension>
using row_vector = Eigen::RowVector<Scalar, N>;

}

#endif
