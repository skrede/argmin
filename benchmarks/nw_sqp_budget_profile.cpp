#include "bench_print.h"
#include "argmin/solver/options.h"
#include "argmin/solver/nw_sqp_policy.h"

#if __has_include("argmin/solver/step_budget_solver.h")
#include "argmin/solver/step_budget_solver.h"
#include "argmin/solver/time_budget_solver.h"
#include "argmin/solver/time_budget_options.h"
#define ARGMIN_NW_PROFILE_HAS_BUDGET_DRIVERS 1
#else
#include "argmin/solver/basic_solver.h"
#define ARGMIN_NW_PROFILE_HAS_BUDGET_DRIVERS 0
#endif

#include "argmin/test_functions/problem_class.h"

#include <Eigen/Core>

#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace
{

template <int NX, int Horizon>
struct double_integrator_profile_problem
{
    static_assert(NX % 2 == 0);

    using scalar_type = double;

    static constexpr int n_x = NX;
    static constexpr int n_u = NX / 2;
    static constexpr int horizon = Horizon;
    static constexpr int n_stage = n_x + n_u;
    static constexpr int problem_dimension = Horizon * n_stage;
    static constexpr int constraint_count = Horizon * n_x;
    static constexpr argmin::problem_class pclass =
          argmin::problem_class::equality
        | argmin::problem_class::bound_constrained
        | argmin::problem_class::application;

    static constexpr scalar_type dt = 0.1;
    static constexpr scalar_type q_diag = 1.0;
    static constexpr scalar_type q_terminal_diag = 5.0;
    static constexpr scalar_type r_diag = 0.05;
    static constexpr scalar_type u_min_scalar = -3.0;
    static constexpr scalar_type u_max_scalar = 3.0;

    [[nodiscard]] static constexpr int x_offset(int k) noexcept
    {
        return k * n_stage;
    }

    [[nodiscard]] static constexpr int u_offset(int k) noexcept
    {
        return k * n_stage + n_x;
    }

    [[nodiscard]] static Eigen::Matrix<scalar_type, n_x, 1> x0_fixed() noexcept
    {
        Eigen::Matrix<scalar_type, n_x, 1> x0;
        x0.setZero();
        for(int axis = 0; axis < n_u; ++axis)
        {
            const scalar_type sign = (axis % 2 == 0) ? 1.0 : -1.0;
            x0[axis] = sign * (1.0 + 0.25 * static_cast<scalar_type>(axis));
            x0[n_u + axis] = sign * (0.15 + 0.05 * static_cast<scalar_type>(axis));
        }
        return x0;
    }

    [[nodiscard]] static Eigen::Matrix<scalar_type, n_x, n_x> A_matrix() noexcept
    {
        Eigen::Matrix<scalar_type, n_x, n_x> A;
        A.setIdentity();
        for(int axis = 0; axis < n_u; ++axis)
            A(axis, n_u + axis) = dt;
        return A;
    }

    [[nodiscard]] static Eigen::Matrix<scalar_type, n_x, n_u> B_matrix() noexcept
    {
        Eigen::Matrix<scalar_type, n_x, n_u> B;
        B.setZero();
        const scalar_type half_dt2 = 0.5 * dt * dt;
        for(int axis = 0; axis < n_u; ++axis)
        {
            B(axis, axis) = half_dt2;
            B(n_u + axis, axis) = dt;
        }
        return B;
    }

    [[nodiscard]] int dimension() const { return problem_dimension; }
    [[nodiscard]] int num_equality() const { return constraint_count; }
    [[nodiscard]] int num_inequality() const { return 0; }

    [[nodiscard]] scalar_type value(
        const Eigen::Vector<scalar_type, problem_dimension>& z) const
    {
        scalar_type cost = 0.5 * q_diag * x0_fixed().squaredNorm();
        for(int k = 0; k < Horizon; ++k)
        {
            const int x_off = x_offset(k);
            const scalar_type q = (k + 1 == Horizon) ? q_terminal_diag : q_diag;
            for(int j = 0; j < n_x; ++j)
                cost += 0.5 * q * z[x_off + j] * z[x_off + j];

            const int u_off = u_offset(k);
            for(int j = 0; j < n_u; ++j)
                cost += 0.5 * r_diag * z[u_off + j] * z[u_off + j];
        }
        return cost;
    }

    void gradient(const Eigen::Vector<scalar_type, problem_dimension>& z,
                  Eigen::Vector<scalar_type, problem_dimension>& g) const
    {
        g.setZero();
        for(int k = 0; k < Horizon; ++k)
        {
            const int x_off = x_offset(k);
            const scalar_type q = (k + 1 == Horizon) ? q_terminal_diag : q_diag;
            for(int j = 0; j < n_x; ++j)
                g[x_off + j] = q * z[x_off + j];

            const int u_off = u_offset(k);
            for(int j = 0; j < n_u; ++j)
                g[u_off + j] = r_diag * z[u_off + j];
        }
    }

    void constraints(const Eigen::Vector<scalar_type, problem_dimension>& z,
                     auto& c) const
    {
        const auto A = A_matrix();
        const auto B = B_matrix();
        const auto x0 = x0_fixed();
        for(int k = 0; k < Horizon; ++k)
        {
            Eigen::Matrix<scalar_type, n_x, 1> x_k;
            if(k == 0)
                x_k = x0;
            else
                for(int j = 0; j < n_x; ++j)
                    x_k[j] = z[x_offset(k - 1) + j];

            Eigen::Matrix<scalar_type, n_u, 1> u_k;
            for(int j = 0; j < n_u; ++j)
                u_k[j] = z[u_offset(k) + j];

            Eigen::Matrix<scalar_type, n_x, 1> x_kp1;
            for(int j = 0; j < n_x; ++j)
                x_kp1[j] = z[x_offset(k) + j];

            const Eigen::Matrix<scalar_type, n_x, 1> residual = x_kp1 - A * x_k - B * u_k;
            const int c_off = k * n_x;
            for(int j = 0; j < n_x; ++j)
                c[c_off + j] = residual[j];
        }
    }

    void constraint_jacobian(
        const Eigen::Vector<scalar_type, problem_dimension>&,
        auto& J) const
    {
        const auto A = A_matrix();
        const auto B = B_matrix();
        J.setZero();

        for(int k = 0; k < Horizon; ++k)
        {
            const int row0 = k * n_x;
            const int col_xkp1 = x_offset(k);
            for(int j = 0; j < n_x; ++j)
                J(row0 + j, col_xkp1 + j) = 1.0;

            const int col_uk = u_offset(k);
            for(int i = 0; i < n_x; ++i)
                for(int j = 0; j < n_u; ++j)
                    J(row0 + i, col_uk + j) = -B(i, j);

            if(k > 0)
            {
                const int col_xk = x_offset(k - 1);
                for(int i = 0; i < n_x; ++i)
                    for(int j = 0; j < n_x; ++j)
                        J(row0 + i, col_xk + j) = -A(i, j);
            }
        }
    }

    [[nodiscard]] Eigen::Vector<scalar_type, problem_dimension> lower_bounds() const
    {
        Eigen::Vector<scalar_type, problem_dimension> lb;
        lb.setConstant(-std::numeric_limits<scalar_type>::infinity());
        for(int k = 0; k < Horizon; ++k)
            for(int j = 0; j < n_u; ++j)
                lb[u_offset(k) + j] = u_min_scalar;
        return lb;
    }

    [[nodiscard]] Eigen::Vector<scalar_type, problem_dimension> upper_bounds() const
    {
        Eigen::Vector<scalar_type, problem_dimension> ub;
        ub.setConstant(std::numeric_limits<scalar_type>::infinity());
        for(int k = 0; k < Horizon; ++k)
            for(int j = 0; j < n_u; ++j)
                ub[u_offset(k) + j] = u_max_scalar;
        return ub;
    }

    [[nodiscard]] Eigen::Vector<scalar_type, problem_dimension> initial_point() const
    {
        const auto A = A_matrix();
        Eigen::Vector<scalar_type, problem_dimension> z;
        z.setZero();

        Eigen::Matrix<scalar_type, n_x, 1> x = x0_fixed();
        for(int k = 0; k < Horizon; ++k)
        {
            x = A * x;
            for(int j = 0; j < n_x; ++j)
                z[x_offset(k) + j] = x[j];
        }
        return z;
    }

    [[nodiscard]] scalar_type optimal_value() const { return 0.0; }
};

struct cli_options
{
    std::string cases{"double_integrator_nx4_n20,double_integrator_nx8_n20"};
    std::string output{};
    std::string library_rev{"working-tree"};
    std::string harness_hash{"unknown"};
    std::string compiler{"unknown"};
    std::string build_type{"unknown"};
    std::string notes{};
    int repetitions{1};
    std::uint32_t step_iterations{12};
    long time_budget_ms{100};
};

struct profile_row
{
    std::string_view case_name;
    std::string_view driver_path;
    std::string status;
    std::string notes;
    int repetition{};
    std::uint32_t iterations{};
    double solve_wall_us{};
    double per_iteration_us{};
    double objective{};
    double constraint_violation{};
};

[[nodiscard]] const char* status_name(argmin::solver_status status)
{
    switch(status)
    {
    case argmin::solver_status::running: return "running";
    case argmin::solver_status::converged: return "converged";
    case argmin::solver_status::max_iterations: return "max_iterations";
    case argmin::solver_status::budget_exhausted: return "budget_exhausted";
    case argmin::solver_status::stalled: return "stalled";
    case argmin::solver_status::diverged: return "diverged";
    case argmin::solver_status::xtol_reached: return "xtol_reached";
    case argmin::solver_status::ftol_reached: return "ftol_reached";
    case argmin::solver_status::maxeval_reached: return "maxeval_reached";
    case argmin::solver_status::roundoff_limited: return "roundoff_limited";
    case argmin::solver_status::trust_region_step_rejected:
        return "trust_region_step_rejected";
    case argmin::solver_status::objective_stalled: return "objective_stalled";
    case argmin::solver_status::time_limit_reached: return "time_limit_reached";
    case argmin::solver_status::aborted: return "aborted";
#if ARGMIN_NW_PROFILE_HAS_BUDGET_DRIVERS
    case argmin::solver_status::invalid_problem: return "invalid_problem";
#endif
    }
    return "unknown";
}

[[nodiscard]] bool case_selected(std::string_view cases, std::string_view name)
{
    if(cases == "all")
        return true;

    while(!cases.empty())
    {
        const std::size_t comma = cases.find(',');
        const std::string_view token = cases.substr(0, comma);
        if(token == name)
            return true;
        if(comma == std::string_view::npos)
            break;
        cases.remove_prefix(comma + 1);
    }
    return false;
}

[[nodiscard]] std::uint32_t parse_u32(const char* text, std::uint32_t fallback)
{
    if(text == nullptr || *text == '\0')
        return fallback;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if(end == text || *end != '\0')
        return fallback;
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] int parse_int(const char* text, int fallback)
{
    if(text == nullptr || *text == '\0')
        return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if(end == text || *end != '\0')
        return fallback;
    return static_cast<int>(value);
}

[[nodiscard]] long parse_long(const char* text, long fallback)
{
    if(text == nullptr || *text == '\0')
        return fallback;
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if(end == text || *end != '\0')
        return fallback;
    return value;
}

void print_usage(const char* argv0)
{
    argmin::bench::println(stderr,
        "usage: {} [--cases names] [--repetitions n] [--output path] "
        "[--library-rev rev] [--harness-hash hash] [--compiler id] "
        "[--build-type type] [--notes text] [--step-iterations n] "
        "[--time-budget-ms n]",
        argv0);
}

[[nodiscard]] bool parse_cli(int argc, char** argv, cli_options& opts)
{
    for(int i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];
        auto next = [&]() -> const char*
        {
            if(i + 1 >= argc)
                return nullptr;
            ++i;
            return argv[i];
        };

        if(std::strcmp(arg, "--cases") == 0)
            opts.cases = next() == nullptr ? opts.cases : argv[i];
        else if(std::strcmp(arg, "--repetitions") == 0)
            opts.repetitions = parse_int(next(), opts.repetitions);
        else if(std::strcmp(arg, "--output") == 0)
            opts.output = next() == nullptr ? opts.output : argv[i];
        else if(std::strcmp(arg, "--library-rev") == 0)
            opts.library_rev = next() == nullptr ? opts.library_rev : argv[i];
        else if(std::strcmp(arg, "--harness-hash") == 0)
            opts.harness_hash = next() == nullptr ? opts.harness_hash : argv[i];
        else if(std::strcmp(arg, "--compiler") == 0)
            opts.compiler = next() == nullptr ? opts.compiler : argv[i];
        else if(std::strcmp(arg, "--build-type") == 0)
            opts.build_type = next() == nullptr ? opts.build_type : argv[i];
        else if(std::strcmp(arg, "--notes") == 0)
            opts.notes = next() == nullptr ? opts.notes : argv[i];
        else if(std::strcmp(arg, "--step-iterations") == 0)
            opts.step_iterations = parse_u32(next(), opts.step_iterations);
        else if(std::strcmp(arg, "--time-budget-ms") == 0)
            opts.time_budget_ms = parse_long(next(), opts.time_budget_ms);
        else if(std::strcmp(arg, "--help") == 0)
        {
            print_usage(argv[0]);
            return false;
        }
        else
        {
            argmin::bench::println(stderr, "unknown argument: {}", arg);
            print_usage(argv[0]);
            return false;
        }
    }

    if(opts.repetitions < 1)
        opts.repetitions = 1;
    if(opts.step_iterations == 0)
        opts.step_iterations = 1;
    if(opts.time_budget_ms < 1)
        opts.time_budget_ms = 1;
    return true;
}

template <typename Problem>
[[nodiscard]] argmin::solver_options<> make_core_options(const cli_options& opts)
{
    argmin::solver_options<> core_opts;
    core_opts.max_iterations = opts.step_iterations;
    core_opts.constraint_tolerance = 1e-8;
    core_opts.feasibility_tolerance = 1e-8;
    core_opts.set_gradient_threshold(1e-8);
    core_opts.set_objective_threshold(1e-12);
    core_opts.set_step_threshold(1e-12);
    core_opts.set_stall_threshold(1e-14);
    core_opts.set_stall_window(50);
    return core_opts;
}

[[nodiscard]] double elapsed_us(std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point stop)
{
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return static_cast<double>(ns) / 1000.0;
}

[[nodiscard]] double per_iteration(double wall_us, std::uint32_t iterations)
{
    if(iterations == 0)
        return std::numeric_limits<double>::infinity();
    return wall_us / static_cast<double>(iterations);
}

template <typename Problem>
[[nodiscard]] profile_row run_step_case(std::string_view case_name,
                                        int repetition,
                                        const cli_options& opts)
{
    Problem problem;
    auto x0 = problem.initial_point();
    auto core_opts = make_core_options<Problem>(opts);

    argmin::nw_sqp_policy<Problem::problem_dimension> policy;
    policy.options.multiplier_reest_every_k = 1;

#if ARGMIN_NW_PROFILE_HAS_BUDGET_DRIVERS
    using solver_type =
        argmin::step_budget_solver<argmin::nw_sqp_policy<Problem::problem_dimension>,
                                   Problem::problem_dimension,
                                   Problem>;
#else
    using solver_type =
        argmin::basic_solver<argmin::nw_sqp_policy<Problem::problem_dimension>,
                             Problem::problem_dimension,
                             Problem>;
#endif
    auto solver = std::make_unique<solver_type>(policy, problem, x0, core_opts);
    const auto start = std::chrono::steady_clock::now();
    const auto result = solver->solve();
    const auto stop = std::chrono::steady_clock::now();
    const double wall_us = elapsed_us(start, stop);

    return profile_row{
        case_name,
        "step_budget_solver",
        status_name(result.status),
        opts.notes,
        repetition,
        result.iterations,
        wall_us,
        per_iteration(wall_us, result.iterations),
        result.objective_value,
        result.constraint_violation,
    };
}

template <typename Problem>
[[nodiscard]] profile_row run_time_case(std::string_view case_name,
                                        int repetition,
                                        const cli_options& opts)
{
    Problem problem;
    auto x0 = problem.initial_point();
    argmin::nw_sqp_policy<Problem::problem_dimension> policy;
    policy.options.multiplier_reest_every_k = 1;

#if ARGMIN_NW_PROFILE_HAS_BUDGET_DRIVERS
    argmin::time_budget_options<> time_opts;
    time_opts.core = make_core_options<Problem>(opts);
    time_opts.core.max_iterations = opts.step_iterations * 100;
    time_opts.max_time = std::chrono::milliseconds(opts.time_budget_ms);
    time_opts.time_poll_stride = 1;

    using solver_type =
        argmin::time_budget_solver<argmin::nw_sqp_policy<Problem::problem_dimension>,
                                   Problem::problem_dimension,
                                   Problem>;
#else
    auto time_opts = make_core_options<Problem>(opts);
    time_opts.max_iterations = opts.step_iterations * 100;
    if constexpr(requires(argmin::solver_options<>& candidate)
                 { candidate.max_time = std::chrono::milliseconds{}; })
        time_opts.max_time = std::chrono::milliseconds(opts.time_budget_ms);

    using solver_type =
        argmin::basic_solver<argmin::nw_sqp_policy<Problem::problem_dimension>,
                             Problem::problem_dimension,
                             Problem>;
#endif
    auto solver = std::make_unique<solver_type>(policy, problem, x0, time_opts);
    const auto start = std::chrono::steady_clock::now();
    const auto result = solver->solve();
    const auto stop = std::chrono::steady_clock::now();
    const double wall_us = elapsed_us(start, stop);

    return profile_row{
        case_name,
        "time_budget_solver",
        status_name(result.status),
        opts.notes,
        repetition,
        result.iterations,
        wall_us,
        per_iteration(wall_us, result.iterations),
        result.objective_value,
        result.constraint_violation,
    };
}

void write_csv_text(std::ostream& out, std::string_view text)
{
    out << '"';
    for(char ch : text)
    {
        if(ch == '"')
            out << "\"\"";
        else
            out << ch;
    }
    out << '"';
}

void write_header(std::ostream& out)
{
    out << "library_rev,harness_hash,compiler,build_type,case,driver_path,"
           "repetition,iterations,solve_wall_us,per_iteration_us,objective,"
           "constraint_violation,status,notes\n";
}

void write_row(std::ostream& out, const cli_options& opts, const profile_row& row)
{
    write_csv_text(out, opts.library_rev);
    out << ',';
    write_csv_text(out, opts.harness_hash);
    out << ',';
    write_csv_text(out, opts.compiler);
    out << ',';
    write_csv_text(out, opts.build_type);
    out << ',';
    write_csv_text(out, row.case_name);
    out << ',';
    write_csv_text(out, row.driver_path);
    out << ',' << row.repetition
        << ',' << row.iterations
        << ',' << row.solve_wall_us
        << ',' << row.per_iteration_us
        << ',' << row.objective
        << ',' << row.constraint_violation
        << ',';
    write_csv_text(out, row.status);
    out << ',';
    write_csv_text(out, row.notes);
    out << '\n';
}

template <typename Problem>
void run_selected_case(std::ostream& out,
                       std::string_view case_name,
                       const cli_options& opts)
{
    if(!case_selected(opts.cases, case_name))
        return;

    for(int rep = 0; rep < opts.repetitions; ++rep)
    {
        write_row(out, opts, run_step_case<Problem>(case_name, rep, opts));
        write_row(out, opts, run_time_case<Problem>(case_name, rep, opts));
    }
}

}

int main(int argc, char** argv)
{
    cli_options opts;
    if(!parse_cli(argc, argv, opts))
        return 1;

    std::ofstream file_out;
    std::ostream* out = &std::cout;
    if(!opts.output.empty())
    {
        file_out.open(opts.output);
        if(!file_out)
        {
            std::fprintf(stderr, "failed to open output path: %s\n", opts.output.c_str());
            return 2;
        }
        out = &file_out;
    }

    *out << std::setprecision(17);
    write_header(*out);
    run_selected_case<double_integrator_profile_problem<4, 20>>(
        *out, "double_integrator_nx4_n20", opts);
    run_selected_case<double_integrator_profile_problem<8, 20>>(
        *out, "double_integrator_nx8_n20", opts);
    return 0;
}
