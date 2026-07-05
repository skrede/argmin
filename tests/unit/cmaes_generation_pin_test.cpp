// CMA-ES single-generation oracle pin.
//
// Pins the state of the production CMA-ES after exactly ONE generation
// against hand-derived golden values (tests/oracles/cmaes_one_generation.csv).
// The oracle is derived from Hansen (2016), "The CMA Evolution Strategy:
// A Tutorial", arXiv:1604.00772, eqs. 41-47, applied to the recorded
// offspring draws at the pinned seed, with the pycma purecma.tell update
// ordering: the covariance C is adapted using the SAMPLING sigma (the
// rank-mu vectors are y_i = (x_(i:lambda) - mean_old) / sigma_old), and
// sigma is adapted AFTER C.
//
// Configuration: n = 2, unbounded shifted sphere
// f(x) = (x1 - 0.5)^2 + 2 (x2 + 0.25)^2, x0 = mean0 = (0, 0),
// sigma0 = 0.5 (explicit), seed = 42, defaults lambda = 6, mu = 3,
// C0 = I, B0 = I, D0 = 1, p_sigma0 = p_c0 = 0.
//
// Derivation walkthrough (numbers cross-checked with an independent
// replay of the tutorial equations; strategy constants below match the
// Hansen sec. 3 defaults exactly):
//   weights (positive, normalized)
//     w = (0.63704257124121677, 0.28457025743803294, 0.07838717132075033)
//   mu_eff = 2.0286114646100617, c_sigma = 0.44620498737831715,
//   d_sigma = 1.4462049873783172, c_c = 0.62455453902682645,
//   c_1 = 0.15481539989641360, c_mu = 0.057859085071916304,
//   chi_n = 1.2542727428189950.
//   Fitness ranking of the six recorded offspring (best first): 3,2,1,0,5,4.
//   mean_new = sum_i w_i x_(i)            (eq. 41-42)
//            = (0.31896918124478485, -1.1936938228168659e-04)
//   delta_w  = (mean_new - mean_old)/sigma0
//            = (0.63793836248956970, -2.3873876456337317e-04)
//   p_sigma  = sqrt(c_sigma (2 - c_sigma) mu_eff) delta_w    (eq. 43, C = I)
//            = 1.1859495... * delta_w
//            = (0.75655800135135487, -2.8313036679325021e-04),
//     |p_sigma| = 0.75655805432997760
//   sigma_1  = sigma0 exp((c_sigma/d_sigma)(|p_sigma|/chi_n - 1))   (eq. 44)
//            = 0.5 exp(0.3085352 * (0.6031843 - 1)) = 0.44238327112618164
//   h_sigma  = 1  (|p_sigma|/sqrt(1-(1-c_sigma)^2) = 0.9088 <
//                  (1.4 + 2/3) chi_n = 2.5922)
//   p_c      = h_sigma sqrt(c_c (2 - c_c) mu_eff) delta_w    (eq. 45)
//            = (0.84214142098174083, -3.1515866462116323e-04)
//   C_1      = (1 - c_1 - c_mu) I + c_1 p_c p_c^T
//              + c_mu sum_i w_i y_i y_i^T                    (eq. 47)
//     with y_i = (x_(i) - mean_old)/sigma0  -- the SAMPLING sigma --
//            = [ 0.94509326764524815  0.0035883043708415784
//                0.0035883043708415784  0.78786940316315690 ]
//
// Red-state proof (pre-fix): the production policy multiplies sigma by
// its adaptation factor BEFORE forming the rank-mu deltas, so the
// deltas are divided by the UPDATED sigma and the rank-mu block is
// mis-scaled by (sigma0/sigma_1)^2 = 1.27745. The observed pre-fix
// covariance is
//   [ 0.95840300940057721  0.0045952658311207133
//     0.0045952658311207133  0.78802029859005884 ]
// (max entry error vs the oracle: 1.33e-2). The mean / p_sigma / p_c /
// sigma legs use the old sigma correctly and stay green, isolating the
// defect to the rank-mu scaling.
//
// Premise guard: the offspring recorded in the CSV are re-sampled here
// from a fresh generator at the same seed and compared before the
// policy runs. The Gaussian sampler keeps a thread-local half-pair
// cache, so this case must run in its own process (the ctest default);
// a premise failure below means the draw stream changed (sampler or
// seeding change, or a pre-populated pair cache), NOT that the update
// equations regressed -- re-record the CSV per the recipe above.

#include "argmin/detail/xoshiro256.h"
#include "argmin/detail/cmaes_sampling.h"
#include "argmin/solver/options.h"
#include "argmin/solver/cmaes_policy.h"
#include "argmin/types.h"

#include <Eigen/Core>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using Catch::Approx;
using namespace argmin;

namespace
{

struct shifted_sphere
{
    static constexpr int problem_dimension = dynamic_dimension;

    int dimension() const { return 2; }

    double value(const Eigen::VectorXd& x) const
    {
        const double a = x[0] - 0.5;
        const double b = x[1] + 0.25;
        return a * a + 2.0 * b * b;
    }
};

// Minimal oracle CSV reader: '#' lines are comments; data lines are
// "name,v1,v2,...". Returns name -> values.
std::map<std::string, std::vector<double>> load_oracle(const std::string& path)
{
    std::map<std::string, std::vector<double>> rows;
    std::ifstream in(path);
    if(!in.is_open())
        return rows;
    std::string line;
    while(std::getline(in, line))
    {
        if(line.empty() || line.front() == '#')
            continue;
        std::istringstream ss(line);
        std::string name;
        std::getline(ss, name, ',');
        std::vector<double> vals;
        std::string tok;
        while(std::getline(ss, tok, ','))
            vals.push_back(std::stod(tok));
        rows.emplace(name, std::move(vals));
    }
    return rows;
}

}

TEST_CASE("cmaes: one hand-derived generation pins mean, p_sigma, p_c, and sigma",
          "[cmaes][oracle-pin]")
{
    const auto oracle = load_oracle("oracles/cmaes_one_generation.csv");
    REQUIRE(oracle.contains("seed"));

    const auto seed = static_cast<std::uint64_t>(oracle.at("seed")[0]);
    const int lambda = static_cast<int>(oracle.at("lambda")[0]);
    const double sigma0 = oracle.at("sigma0")[0];
    const int n = static_cast<int>(oracle.at("n")[0]);
    REQUIRE(n == 2);
    REQUIRE(lambda == 6);

    // Premise: the sampler at the pinned seed reproduces the recorded
    // offspring (identity B, unit D, mean0 = 0). lambda * n = 12 is even,
    // so this capture leaves the Marsaglia half-pair cache empty and the
    // policy below re-draws the identical stream from its own seeded rng.
    Eigen::VectorXd mean0(2);
    mean0 << oracle.at("mean0")[0], oracle.at("mean0")[1];
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd D = Eigen::VectorXd::Ones(2);
    detail::xoshiro256 rng{seed};
    const auto offspring = detail::sample_offspring<double, Eigen::Dynamic>(
        mean0, sigma0, B, D, lambda, rng);
    for(int i = 0; i < lambda; ++i)
    {
        const auto& col = oracle.at("offspring_" + std::to_string(i));
        INFO("offspring premise (draw stream changed? see header comment), "
             "column " << i);
        REQUIRE(offspring(0, i) == Approx(col[0]).epsilon(0).margin(1e-15));
        REQUIRE(offspring(1, i) == Approx(col[1]).epsilon(0).margin(1e-15));
    }

    // One production generation.
    shifted_sphere problem;
    Eigen::VectorXd x0 = mean0;
    solver_options opts;
    cmaes_policy<> policy;
    policy.options.seed = seed;
    policy.options.initial_sigma = sigma0;
    auto s = policy.init(problem, x0, opts);
    (void)policy.step(s);

    // Green legs: mean, p_sigma, p_c, sigma all use the sampling sigma
    // correctly in the current implementation and must match the oracle.
    // This case stays untagged: it must keep passing as a regression
    // guard on these four quantities regardless of the rank-mu C fix
    // pinned separately below.
    const auto& mean = oracle.at("mean");
    CHECK(s.mean[0] == Approx(mean[0]).epsilon(0).margin(1e-12));
    CHECK(s.mean[1] == Approx(mean[1]).epsilon(0).margin(1e-12));

    const auto& p_sigma = oracle.at("p_sigma");
    CHECK(s.p_sigma[0] == Approx(p_sigma[0]).epsilon(0).margin(1e-12));
    CHECK(s.p_sigma[1] == Approx(p_sigma[1]).epsilon(0).margin(1e-12));

    const auto& p_c = oracle.at("p_c");
    CHECK(s.p_c[0] == Approx(p_c[0]).epsilon(0).margin(1e-12));
    CHECK(s.p_c[1] == Approx(p_c[1]).epsilon(0).margin(1e-12));

    CHECK(s.sigma == Approx(oracle.at("sigma")[0]).epsilon(0).margin(1e-12));
}

// RED against the current substrate: the production policy multiplies
// sigma by its adaptation factor BEFORE forming the rank-mu deltas, so
// the deltas are divided by the UPDATED sigma and the rank-mu block is
// mis-scaled by (sigma0/sigma_1)^2 = 1.27745 (see header derivation).
// [!shouldfail] records this as the expected disposition; once the
// rank-mu update is reordered to use the sampling sigma, this case
// starts passing and the tag must be removed.
TEST_CASE("cmaes: one hand-derived generation pins the rank-mu covariance C",
          "[cmaes][oracle-pin][!shouldfail]")
{
    const auto oracle = load_oracle("oracles/cmaes_one_generation.csv");
    REQUIRE(oracle.contains("seed"));

    const auto seed = static_cast<std::uint64_t>(oracle.at("seed")[0]);
    const int lambda = static_cast<int>(oracle.at("lambda")[0]);
    const double sigma0 = oracle.at("sigma0")[0];
    const int n = static_cast<int>(oracle.at("n")[0]);
    REQUIRE(n == 2);
    REQUIRE(lambda == 6);

    // Premise: identical re-draw check as the green-leg case above.
    Eigen::VectorXd mean0(2);
    mean0 << oracle.at("mean0")[0], oracle.at("mean0")[1];
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd D = Eigen::VectorXd::Ones(2);
    detail::xoshiro256 rng{seed};
    const auto offspring = detail::sample_offspring<double, Eigen::Dynamic>(
        mean0, sigma0, B, D, lambda, rng);
    for(int i = 0; i < lambda; ++i)
    {
        const auto& col = oracle.at("offspring_" + std::to_string(i));
        INFO("offspring premise (draw stream changed? see header comment), "
             "column " << i);
        REQUIRE(offspring(0, i) == Approx(col[0]).epsilon(0).margin(1e-15));
        REQUIRE(offspring(1, i) == Approx(col[1]).epsilon(0).margin(1e-15));
    }

    // One production generation.
    shifted_sphere problem;
    Eigen::VectorXd x0 = mean0;
    solver_options opts;
    cmaes_policy<> policy;
    policy.options.seed = seed;
    policy.options.initial_sigma = sigma0;
    auto s = policy.init(problem, x0, opts);
    (void)policy.step(s);

    // Covariance pin (Hansen eq. 47): the rank-mu deltas MUST use the
    // sampling sigma. An implementation that divides by the updated
    // sigma mis-scales the rank-mu block by (sigma0/sigma_1)^2 = 1.27745
    // and fails this leg by ~1.3e-2 in the (0,0) entry.
    const auto& C = oracle.at("C");
    INFO("C(0,0) observed " << s.C(0, 0) << " pinned " << C[0]);
    INFO("C(0,1) observed " << s.C(0, 1) << " pinned " << C[1]);
    INFO("C(1,1) observed " << s.C(1, 1) << " pinned " << C[3]);
    CHECK(s.C(0, 0) == Approx(C[0]).epsilon(0).margin(1e-12));
    CHECK(s.C(0, 1) == Approx(C[1]).epsilon(0).margin(1e-12));
    CHECK(s.C(1, 0) == Approx(C[2]).epsilon(0).margin(1e-12));
    CHECK(s.C(1, 1) == Approx(C[3]).epsilon(0).margin(1e-12));
}
