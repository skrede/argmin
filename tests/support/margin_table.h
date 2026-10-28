#ifndef HPP_GUARD_ARGMIN_TESTS_SUPPORT_MARGIN_TABLE_H
#define HPP_GUARD_ARGMIN_TESTS_SUPPORT_MARGIN_TABLE_H

#include "cases/case_options.h"

#include "oracle/result_oracle.h"

#include "argmin/result/status.h"

#include <string>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <string_view>

namespace argmin::cases
{

// With ARGMIN_ORACLE_MARGIN_DIR unset the table writes nothing; ratio is the leg over its unscaled
// tolerance, so it reads directly as the safety factor the case would need.
class margin_table
{
public:
    explicit margin_table(std::string_view family)
    {
        const char* dir = std::getenv("ARGMIN_ORACLE_MARGIN_DIR");
        if(dir == nullptr)
            return;
        std::filesystem::create_directories(dir);
        out_.open(std::filesystem::path(dir) / (std::string(family) + ".csv"), std::ios::trunc);
        out_ << "key,class,problem,status,leg,value,tolerance,ratio\n";
    }

    template <typename Scalar>
    void record(const case_id& id, solver_status status, const oracle::verdict<Scalar>& v, const oracle::tolerances<Scalar>& tol)
    {
        if(!out_.is_open())
            return;
        const Scalar stationarity = v.claimed == oracle::claim::ftol_optimality ? tol.ftol_stationarity : tol.stationarity;
        row(id, status, "stationarity", v.legs.stationarity, stationarity);
        row(id, status, "equality_violation", v.legs.equality_violation, tol.feasibility);
        row(id, status, "inequality_violation", v.legs.inequality_violation, tol.feasibility);
        row(id, status, "bound_violation", v.legs.bound_violation, tol.feasibility);
        row(id, status, "complementarity", v.legs.complementarity, stationarity);
        row(id, status, "dual_sign_breach", v.legs.dual_sign_breach, stationarity);
    }

private:
    std::ofstream out_;

    template <typename Scalar>
    void row(const case_id& id, solver_status status, std::string_view leg, Scalar value, Scalar tolerance)
    {
        out_ << id.key << ',' << id.problem_class << ',' << id.problem << ',' << static_cast<int>(status) << ',' << leg << ',' << std::scientific << std::setprecision(6) << value << ',' << tolerance << ','
             << value / tolerance << '\n';
    }
};

}

#endif
