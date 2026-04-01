#ifndef HPP_GUARD_NABLAPP_OPTIONS_BFGS_OPTIONS_H
#define HPP_GUARD_NABLAPP_OPTIONS_BFGS_OPTIONS_H

#include <optional>

namespace nablapp
{

// BFGS Hessian approximation parameters.
// Reference: N&W 2e, Procedure 18.2.
struct bfgs_options
{
    std::optional<double> damping_threshold{};  // Powell damping sTBs threshold (default: 0.2, N&W Proc 18.2)
    std::optional<double> damping_factor{};     // Powell damping blend factor (default: 0.8, N&W Proc 18.2)
};

}

#endif
