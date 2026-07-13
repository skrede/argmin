// CMA-ES single-generation oracle pin.
//
// Pins the state of the production CMA-ES after exactly ONE generation
// against hand-derived golden values (tests/oracles/cmaes_one_generation.csv).
// The oracle is derived from Hansen (2016), "The CMA Evolution Strategy:
// A Tutorial", arXiv:1604.00772, eqs. 41-47, applied to the offspring the
// draw path produces at the pinned seed, with the pycma purecma.tell update
// ordering: the covariance C is adapted using the SAMPLING sigma (the
// rank-mu vectors are y_i = (x_(i:lambda) - mean_old) / sigma_old), and
// sigma is adapted AFTER C. The offspring themselves are NOT captured from
// the library sampler: the CSV rows come from an independent offline replay
// of the engine and mapping in exact arithmetic with a correctly-rounded
// log reference (see the CSV header), so this pin stays an independent
// check rather than a self-referential gate.
//
// Configuration: n = 2, unbounded shifted sphere
// f(x) = (x1 - 0.5)^2 + 2 (x2 + 0.25)^2, x0 = mean0 = (0, 0),
// sigma0 = 0.5 (explicit), seed = 42, defaults lambda = 6, mu = 3,
// C0 = I, B0 = I, D0 = 1, p_sigma0 = p_c0 = 0.
//
// Derivation walkthrough (numbers cross-checked with two independent
// numpy formulations of the tutorial equations, which agree bit-for-bit;
// strategy constants below match the Hansen sec. 3 defaults exactly):
//   weights (positive, normalized)
//     w = (0.63704257124121677, 0.28457025743803294, 0.07838717132075033)
//   mu_eff = 2.0286114646100617, c_sigma = 0.44620498737831715,
//   d_sigma = 1.4462049873783172, c_c = 0.62455453902682645,
//   c_1 = 0.15481539989641360, c_mu = 0.057859085071916304,
//   chi_n = 1.2542727428189950.
//   Fitness ranking of the six offspring (best first): 3,2,1,0,5,4. The
//   smallest gap between adjacent sorted fitnesses is 0.0866, so the rank
//   order is far from any tie -- no near-threshold ranking flip.
//   mean_new = sum_i w_i x_(i)            (eq. 41-42)
//            = (0.31896918124478490, -1.1936938228157557e-04)
//   delta_w  = (mean_new - mean_old)/sigma0
//            = (0.63793836248956980, -2.3873876456315113e-04)
//   p_sigma  = sqrt(c_sigma (2 - c_sigma) mu_eff) delta_w    (eq. 43, C = I)
//            = 1.1859495... * delta_w
//            = (0.75655800135135509, -2.8313036679298686e-04),
//     |p_sigma| = 0.75655805432997780
//   sigma_1  = sigma0 exp((c_sigma/d_sigma)(|p_sigma|/chi_n - 1))   (eq. 44)
//            = 0.5 exp(0.3085352 * (0.6031843 - 1)) = 0.44238327112618170
//   h_sigma  = 1  (|p_sigma|/sqrt(1-(1-c_sigma)^2) = 0.90861 <
//                  (1.4 + 2/3) chi_n = 2.59216; slack 1.6836, so the
//                  indicator has wide margin from its threshold)
//   p_c      = h_sigma sqrt(c_c (2 - c_c) mu_eff) delta_w    (eq. 45)
//            = (0.84214142098174094, -3.1515866462087012e-04)
//   C_1      = (1 - c_1 - c_mu) I + c_1 p_c p_c^T
//              + c_mu sum_i w_i y_i y_i^T                    (eq. 47)
//     with y_i = (x_(i) - mean_old)/sigma0  -- the SAMPLING sigma --
//            = [ 0.94509326764524804  0.0035883043708416083
//                0.0035883043708416083  0.78786940316315690 ]
//
// Rank-mu scaling guard: the covariance pin below exists to catch an
// implementation that divides the rank-mu deltas by the UPDATED sigma
// instead of the sampling sigma. Dividing by sigma_1 rather than sigma0
// mis-scales the rank-mu block by (sigma0/sigma_1)^2 = 1.27745, which
// gives the wrong covariance
//   [ 0.95840300940057710  0.0045952658311207400
//     0.0045952658311207400  0.78802029859005880 ]
// (max entry error vs the oracle: 1.33e-2). The mean / p_sigma / p_c /
// sigma legs use the sampling sigma correctly and are unaffected, which
// isolates the guarded fault to the rank-mu scaling.
//
// Premise guard: the offspring in the CSV are re-sampled here from a
// fresh generator at the same seed and compared before the policy runs.
// The Marsaglia polar sampler carries no cross-call state -- each call
// consumes whole polar pairs from the RNG stream and keeps no half-pair
// between calls -- so the same seed reproduces the same offspring stream
// deterministically, and the policy below re-draws that identical stream
// from its own seeded generator. A premise failure means the draw stream
// changed (sampler or seeding change), NOT that the update equations
// regressed -- regenerate the CSV with the independent replay above. The
// uniform mapping is now exact IEEE-754 bit arithmetic, so the offspring
// are bit-identical across platforms by construction; the only residual
// cross-platform variation is the libm log, a few ULP, absorbed by the
// premise margin below.

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

    // Premise: the sampler at the pinned seed reproduces the oracle
    // offspring (identity B, unit D, mean0 = 0). The Marsaglia polar
    // sampler keeps no half-pair state between calls, so the seeded stream
    // fully determines the offspring, and the policy below re-draws the
    // identical stream from its own seeded generator.
    Eigen::VectorXd mean0(2);
    mean0 << oracle.at("mean0")[0], oracle.at("mean0")[1];
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd D = Eigen::VectorXd::Ones(2);
    detail::xoshiro256 rng{seed};
    const auto offspring = detail::sample_offspring<double, Eigen::Dynamic>(
        mean0, sigma0, B, D, lambda, rng);
    // Margin 2e-15 absolute: the uniform mapping is exact IEEE-754 bit
    // arithmetic, so the uniforms are bit-identical across platforms and
    // the only residual variation is the libm log. A +/-k-ULP log error
    // moves an offspring value by <= ~k ULP (measured: 1.11e-16 at
    // +/-1-2 ULP, 2.22e-16 at +/-4 ULP), and offspring magnitudes here are
    // <= 0.6, so 2e-15 is ~4x the conservative cross-libm + FMA-assembly +
    // CSV-rounding envelope while staying ~13 orders of magnitude below the
    // 1.3e-2 rank-mu fault the covariance pin exists to catch.
    for(int i = 0; i < lambda; ++i)
    {
        const auto& col = oracle.at("offspring_" + std::to_string(i));
        INFO("offspring premise (draw stream changed? see header comment), "
             "column " << i);
        REQUIRE(offspring(0, i) == Approx(col[0]).epsilon(0).margin(2e-15));
        REQUIRE(offspring(1, i) == Approx(col[1]).epsilon(0).margin(2e-15));
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

// The rank-mu covariance deltas divide each offspring displacement by
// the sigma that SAMPLED it (the pre-update step size), so the rank-mu
// block of C matches Hansen eq. 47 exactly. An implementation that
// divides by the already-updated sigma mis-scales the rank-mu block by
// (sigma0/sigma_1)^2 = 1.27745 (see header derivation) and fails the
// C(0,0) leg by ~1.3e-2; this case pins against that regression.
TEST_CASE("cmaes: one hand-derived generation pins the rank-mu covariance C",
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

    // Premise: identical re-draw check as the green-leg case above. The
    // 2e-15 margin is justified in that case's premise comment.
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
        REQUIRE(offspring(0, i) == Approx(col[0]).epsilon(0).margin(2e-15));
        REQUIRE(offspring(1, i) == Approx(col[1]).epsilon(0).margin(2e-15));
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
