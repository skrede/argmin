#ifndef HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_QP_H
#define HPP_GUARD_ARGMIN_TESTS_PROPERTY_GENERATED_QP_H

#include "generated_values.h"

#include "cases/qp_cases.h"

#include "oracle/qp_oracle.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <rapidcheck.h>

#include <limits>
#include <cstdint>
#include <ostream>
#include <algorithm>

namespace argmin::property
{

enum class row_kind : std::uint8_t
{
    equality,
    two_sided,
    lower_only,
    upper_only
};

// Each row is an equality, a two-sided band, or one-sided around the value it takes at a feasible point,
// so l <= u holds and the feasible set is never empty. The active-set solver's contract admits at most n
// equality rows, so a further equality becomes a band.
inline void pick_rows_around(const Eigen::VectorXd& ax, Eigen::Index n, Eigen::VectorXd& l, Eigen::VectorXd& u)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    l = ax;
    u = ax;
    Eigen::Index equalities = 0;
    for(Eigen::Index i = 0; i < ax.size(); ++i)
    {
        auto kind = static_cast<row_kind>(*rc::gen::inRange<std::int32_t>(0, 4));
        if(kind == row_kind::equality && equalities == n)
            kind = row_kind::two_sided;
        if(kind == row_kind::equality)
        {
            ++equalities;
            continue;
        }
        l[i] = kind == row_kind::upper_only ? -inf : ax[i] - pick_value(0, 1);
        u[i] = kind == row_kind::lower_only ? inf : ax[i] + pick_value(0, 1);
    }
}

inline cases::qp_case pick_convex_qp()
{
    const Eigen::Index n = pick_size(1, 6);
    const Eigen::Index m = pick_size(0, 4);
    cases::qp_case c{pick_spd(n), pick_vector(n), pick_matrix(m, n), Eigen::VectorXd(m), Eigen::VectorXd(m), pick_vector(n)};
    pick_rows_around(c.A * c.start, n, c.l, c.u);
    return c;
}

inline rc::Gen<cases::qp_case> convex_qp()
{
    return rc::gen::exec(&pick_convex_qp);
}

// Moves x by P^-1 s 1, which adds s to every component of the stationarity residual; s is 100 times the
// acceptance bound of the unmoved point, so the oracle must reject the moved one on any problem.
inline qp_result<double> planted_qp_result(const cases::qp_case& c, qp_result<double> result, const oracle::qp_tolerances<double>& tol)
{
    const Eigen::VectorXd x = result.x;
    const double scale = std::max({oracle::inf_norm(c.P * x), oracle::inf_norm(c.A.transpose() * result.y), oracle::inf_norm(c.q)});
    const double shift = 100.0 * oracle::safety_factor * (tol.absolute + tol.relative * scale);
    result.x = x + c.P.ldlt().solve(Eigen::VectorXd::Constant(x.size(), shift));
    return result;
}

}

namespace argmin::cases
{

inline void showValue(const qp_case& c, std::ostream& os)
{
    const Eigen::IOFormat full(Eigen::FullPrecision);
    os << "P\n" << c.P.format(full) << "\nq " << c.q.transpose().format(full) << "\nA\n" << c.A.format(full);
    os << "\nl " << c.l.transpose().format(full) << "\nu " << c.u.transpose().format(full) << "\nstart " << c.start.transpose().format(full);
}

}

#endif
