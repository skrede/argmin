#include "argmin/result/status.h"
#include "argmin/result/step_result.h"

#include "argmin/formulation/concepts.h"

#include <Eigen/Core>

namespace
{

struct void_reset_stepper
{
    using scalar_type = double;
    using state_type = Eigen::VectorXd;

    argmin::step_result<double> step() { return {}; }

    const state_type& state() const { return x; }

    argmin::solver_status status() const { return argmin::solver_status::running; }

    void reset(const Eigen::VectorXd&) {}

    void reset_clear(const Eigen::VectorXd&) {}

    void abort() {}

    double constraint_violation() const { return 0.0; }

    state_type x;
};

}

static_assert(argmin::steppable<void_reset_stepper>, "a steppable solver's resets must report a solver status");
