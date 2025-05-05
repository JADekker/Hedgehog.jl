using Revise, Hedgehog, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using StochasticDiffEq

function test()
    # ------------------------------
    # Define payoff and pricing problem
    # ------------------------------
    strike = 1.0
    expiry = Date(2021, 1, 1)

    euro_payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

    reference_date = Date(2020, 1, 1)
    rate = 0.03
    spot = 1.0
    sigma = 0.04
    lambda = 0.1
    mu_jump = 0.0
    sigma_jump = 0.1

    market_inputs = MertonInputs(reference_date, rate, spot, sigma, lambda, mu_jump, sigma_jump)
    euro_pricing_prob = PricingProblem(euro_payoff, market_inputs)

    dynamics = LognormalDynamics()
    trajectories = 10000
    config = Hedgehog.SimulationConfig(trajectories; steps=100, variance_reduction=Hedgehog.NoVarianceReduction())
    strategy = EulerMaruyama()
    montecarlo_method = MonteCarlo(dynamics, strategy, config)

    solution = Hedgehog.solve(euro_pricing_prob, montecarlo_method).price

    @btime Hedgehog.solve($euro_pricing_prob, $montecarlo_method).price
    @btime Hedgehog.solve($euro_pricing_prob, BlackScholesAnalytic()).price

    fd_method = FiniteDifference(1E-4, Hedgehog.FDForward())
    ad_method = ForwardAD()

    spot_lens = @optic _.market_inputs.spot
    delta_prob = Hedgehog.GreekProblem(euro_pricing_prob, spot_lens)
    solve(delta_prob, ad_method, BlackScholesAnalytic())
    @btime solve(delta_prob, fd_method, montecarlo_method)
    @btime solve(delta_prob, ad_method, montecarlo_method)

    rate_greek_prob = GreekProblem(euro_pricing_prob, ZeroRateSpineLens(1))
    solve(rate_greek_prob, ad_method, montecarlo_method)
    @btime solve($rate_greek_prob, $ad_method, $montecarlo_method)
    @btime solve($rate_greek_prob, $fd_method, $montecarlo_method)
end
test()