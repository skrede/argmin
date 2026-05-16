#ifndef HPP_GUARD_ARGMIN_TYPES_H
#define HPP_GUARD_ARGMIN_TYPES_H

#include <Eigen/Core>

// Force-inline attribute portable across GCC, Clang, and MSVC.
//
// Used on per-iteration helper functions where the function-call boundary
// would otherwise prevent the compiler from optimizing through the
// Eigen::Ref<> conversion at the helper signature. Even at -O3 the call
// boundary stops the optimizer from collapsing the Ref<> wrapper on
// otherwise-trivial scatter / accumulate hot paths; an explicit always-
// inline hint recovers the pre-extraction inline-arithmetic codegen.
//
// Reference: GCC manual "Function Attributes" / always_inline; Clang
//            extends the GNU attribute set; Microsoft __forceinline.
#if defined(__GNUC__) || defined(__clang__)
    #define ARGMIN_FORCE_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
    #define ARGMIN_FORCE_INLINE __forceinline
#else
    #define ARGMIN_FORCE_INLINE inline
#endif

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
