#ifndef HPP_GUARD_ARGMIN_DETAIL_COBYLA_ENGINE_H
#define HPP_GUARD_ARGMIN_DETAIL_COBYLA_ENGINE_H

#include "argmin/types.h"

#include "argmin/result/status.h"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <utility>
#include <functional>

namespace argmin::detail
{

// A single COBYLA solve, driven one evaluation at a time.
//
// The evaluator receives the point to evaluate the user at, which lies in the box, then the
// engine's own point, from which the bound rows are built, and fills the length-m constraint
// vector in Powell's convention (every entry must become non-negative), returning the objective.
template <typename Scalar = double>
class cobyla_engine
{
public:
    using vector_type = Eigen::Vector<Scalar, argmin::dynamic_dimension>;
    using matrix_type = Eigen::Matrix<Scalar, argmin::dynamic_dimension, argmin::dynamic_dimension>;
    using evaluator = std::function<Scalar(const vector_type&, const vector_type&, vector_type&)>;

    cobyla_engine() = default;

    // n      : number of variables.
    // m      : number of Powell constraints (all >= 0 when feasible).
    // x0     : starting point (already projected into the box).
    // lower  : lower bounds (may be -inf).
    // upper  : upper bounds (may be +inf).
    // rhobeg : initial trust radius.
    // rhoend : final trust radius.
    // eval   : (x_eval, x, con_out[m]) -> f. Fills con_out (size m) and returns f.
    void init(int n, int m,
              const vector_type& x0,
              const vector_type& lower,
              const vector_type& upper,
              Scalar rhobeg, Scalar rhoend,
              evaluator eval);

    // Perform one further evaluation cycle. Returns true while iterating and
    // false once the solve has terminated (RHO reached RHOEND, or a
    // rounding-limited breakdown).
    bool advance();

    [[nodiscard]] bool converged() const { return converged_; }
    [[nodiscard]] solver_status status() const { return status_; }
    [[nodiscard]] Scalar rho() const { return rho_; }
    [[nodiscard]] Scalar parmu() const { return parmu_; }
    [[nodiscard]] Scalar objective_value() const { return f_best_; }
    [[nodiscard]] Scalar resmax() const { return resmax_best_; }
    [[nodiscard]] const vector_type& x() const { return x_best_; }
    std::uint32_t clamped_evaluations() const { return clamped_evaluations_; }
    [[nodiscard]] Scalar gradient_proxy() const
    {
        Scalar g = Scalar(0);
        for(int i = 1; i <= n_; ++i)
            g += a_(i, mp_) * a_(i, mp_);
        return std::sqrt(g);
    }

private:
    enum class l550_result { continue_iter, converged };
    enum class l440_action { back_to_main, reduce_rho };
    enum class cycle_outcome { reiterate, stepped, terminated };

    void size_workspace();
    cycle_outcome trust_region_cycle();
    cycle_outcome evaluate_trial();

    std::uint32_t lcg_rand();
    Scalar lcg_urand(Scalar a, Scalar b);

    vector_type origin_coords() const;
    void clamp_into_box(const vector_type& x);
    void evaluate(const vector_type& x);
    void bootstrap(const vector_type& x0);
    void build_initial_simplex(const vector_type& x0);
    Scalar initial_step_length(std::int32_t i, Scalar xi) const;
    void evaluate_initial_vertices();
    void exchange_initial_vertex();

    void identify_optimal();
    std::int32_t select_optimal_vertex() const;
    void move_to_pole(std::int32_t nbest);
    bool check_simi_error();
    void build_models();
    void compute_vsig_veta();

    void geometry_step();
    void select_geometry_vertex();
    Scalar geometry_step_sign() const;
    Scalar fit_geometry_step(Scalar dxsign);
    void fit_coordinate_to_bounds(std::int32_t i);
    void apply_geometry_step(Scalar temp);

    void run_trstlp();
    void clamp_dx_to_bounds();
    void compute_prediction();
    bool parmu_raise_changes_optimal() const;

    l440_action process_trial();
    Scalar trial_reduction();
    void drop_by_ratio(Scalar trured);
    std::int32_t drop_by_edge(Scalar trured) const;
    Scalar trial_edge_length(std::int32_t j) const;
    void replace_dropped_vertex();
    l550_result l550();
    void rederive_parmu();
    std::pair<Scalar, Scalar> row_range(std::int32_t k) const;
    void mirror_best();

    // Problem dimensions.
    int n_{0};
    int m_{0};
    int mp_{0};
    int mpp_{0};
    int np_{0};

    // Reference working storage (1-indexed access, row/col 0 unused).
    matrix_type sim_;
    matrix_type simi_;
    matrix_type datmat_;
    matrix_type a_;
    vector_type con_;
    vector_type vsig_;
    vector_type veta_;
    vector_type sigbar_;
    vector_type w_;

    // 0-indexed working vectors.
    vector_type dx_;
    vector_type xcur_;
    vector_type x_eval_;
    vector_type lower_;
    vector_type upper_;

    // Scalars.
    Scalar rho_{0};
    Scalar rho_initial_{0};
    Scalar rhoend_{0};
    Scalar parmu_{0};
    Scalar parsig_{0};
    Scalar pareta_{0};
    Scalar resmax_{0};
    Scalar f_last_{0};
    Scalar prerec_{0};
    Scalar prerem_{0};
    Scalar sum_{0};

    // Control flags.
    int ibrnch_{0};
    int iflag_{0};
    int jdrop_{0};
    int ifull_{1};
    int nevals_{0};
    std::uint32_t seed_{0};
    std::uint32_t clamped_evaluations_{0};
    bool converged_{false};
    solver_status status_{solver_status::running};

    // Incumbent mirror.
    vector_type x_best_;
    Scalar f_best_{0};
    Scalar resmax_best_{0};

    evaluator eval_{};
};

}

#endif
