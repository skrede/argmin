#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>

#include <bit>
#include <array>
#include <cmath>
#include <tuple>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <utility>
#include <variant>
#include <version>
#include <concepts>
#include <optional>
#include <algorithm>
#include <functional>
#include <string_view>
#include <type_traits>

// The macros windows.h defines without NOMINMAX. libstdc++ is not guarded against them, so the Eigen and standard
// headers come first.
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

#include "argmin/argmin.h"
