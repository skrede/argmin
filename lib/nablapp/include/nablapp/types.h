#ifndef HPP_GUARD_NABLAPP_TYPES_H
#define HPP_GUARD_NABLAPP_TYPES_H

#include <Eigen/Core>

namespace nablapp
{

// Vocabulary types for nablapp.
//
// All concept interfaces and solver code use these aliases. Eigen expression
// templates are lazy -- do NOT use `auto` to capture Eigen expressions as
// temporaries, because the underlying storage may be invalidated before
// evaluation. Always assign to a concrete type (vector<>, matrix<>, etc.).

template <typename Scalar = double>
using vector = Eigen::VectorX<Scalar>;

template <typename Scalar = double>
using matrix = Eigen::MatrixX<Scalar>;

template <typename Scalar = double>
using row_vector = Eigen::RowVectorX<Scalar>;

}

#endif
