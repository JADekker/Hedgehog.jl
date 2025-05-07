using Revise, Hedgehog, BenchmarkTools, Dates, Random, Accessors, Optim
import Hedgehog: AbstractPricingMethod, AbstractPricingSolution, PriceDynamics, SimulationStrategy, SimulationConfig, PricingProblem, sde_problem
includet("src/pricing_methods/andersen.jl")
function test()
    # Define market inputs
    strike = 10.0
    reference_date = Date(2020, 1, 1)
    expiry = reference_date + Year(1)
    rate = 0.05
    spot = 10.0
    sigma = 0.2
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # Define payoff
    american_payoff = VanillaOption(strike, expiry, American(), Put(), Spot())

    # -- Wrap everything into a pricing problem
    prob = PricingProblem(american_payoff, market_inputs)

    # --- LSM using `solve(...)` style
    dynamics = LognormalDynamics()
    trajectories = 10000
    steps_lsm = 100

    strategy = BlackScholesExact()
    config = Hedgehog.SimulationConfig(trajectories; steps=steps_lsm, variance_reduction=Hedgehog.Antithetic())
    degree = 5
    lsm_method = LSM(dynamics, strategy, config, degree)
    lsm_solution = Hedgehog.solve(prob, lsm_method)

    println("LSM American Price:")
    println(lsm_solution.price)

    andersen_method = Andersen(dynamics, AndersenPolicy1(), strategy, config, degree)
    andersen_solution = solve(prob, andersen_method)
    println("Andersen American Price:")
    println(andersen_solution.price)
end
test()