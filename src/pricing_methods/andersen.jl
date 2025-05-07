abstract type AndersenPolicy end
"""
    Andersen <: AbstractPricingMethod

Andersen Approximate Policy pricing method for Bermudan option pricing, based on 
    Andersen (2000): A simple approach to the pricing of Bermudan swaptions in the multi-factor LIBOR market model. 

Uses a time-varying barrier to determine approximate optimal stopping times. 

# Fields
- `mc_method`: A `MonteCarlo` method specifying dynamics and simulation strategy.
- `degree`: Degree of the polynomial basis for regression.
- `policy`: The policy type. Currently, `AndersenPolicy1` and `AndersenPolicy2` corresponding to (9) and (10) in Andersen (2000) are implemented.
"""
struct Andersen{M<:MonteCarlo,T<:AndersenPolicy} <: AbstractPricingMethod
    mc_method::M
    degree::Int  # degree of polynomial basis
    policy::T
end

"""
    AndersenSolution{T <: Number,TEl, P<:PricingProblem, M <: AbstractPricingMethod} <: AbstractPricingSolution

Represents a pricing solution obtained using the Andersen method
typically applied to American-style options via Monte Carlo simulation.

# Fields
- `problem::P`: The pricing problem definition (`<: PricingProblem`), usually involving early exercise features.
- `method::M`: The specific Andersen configuration (`<: AbstractPricingMethod`), which carries the estimated coefficients. 
- `price::T`: The calculated numerical price (`<: Number`).
- `stopping_info::Vector{Tuple{Int,S}}`: Information related to the derived optimal stopping (exercise) rule at different time steps. Often contains time indices and associated data `S` (e.g., regression coefficients, exercise boundaries).
- `spot_paths::Matrix{TEl}`: The matrix of simulated underlying asset price paths used in the Andersen algorithm. `TEl` is the element type of the path values.
"""
struct AndersenSolution{T <: Number, S, TEl, P<:PricingProblem, M <: AbstractPricingMethod} <: AbstractPricingSolution
    problem::P
    method::M
    price::T
    stopping_info::Vector{Tuple{Int,S}}
    spot_paths::Matrix{TEl}
end

""" 
    AndersenPolicy1 <: AndersenPolicy

The first Andersen policy computes a time-varying barrier H. 
"""
struct AndersenPolicy1 <: AndersenPolicy end
struct AndersenPolicy1Est{T} <: AndersenPolicy
    H::Vector{T}
end

""" 
    AndersenPolicy2 <: AndersenPolicy

The second Andersen policy computes a time-varying barrier H but also requires the exercise value to exceed the maximum of the implied future Euroopean option prices. 
"""
struct AndersenPolicy2 <: AndersenPolicy end
struct AndersenPolicy2Est <: AndersenPolicy boundary::Vector{T} end

"""
    Andersen(dynamics::PriceDynamics, strategy::SimulationStrategy, config::SimulationConfig, degree::Int)

Constructs an `Andersen` pricing method with polynomial regression and Monte Carlo simulation.

# Arguments
- `dynamics`: Price dynamics.
- `strategy`: Simulation strategy.
- `config`: Monte Carlo configuration.
- `degree`: Degree of polynomial regression.

# Returns
- An `Andersen` instance.
"""
function Andersen(dynamics::PriceDynamics, policy::AndersenPolicy, strategy::SimulationStrategy, config::SimulationConfig, degree::Int)
    mc = MonteCarlo(dynamics, strategy, config)
    return Andersen(mc, degree, policy)
end

"""
    solve(prob::PricingProblem, method::Andersen)

Prices an American option using the Andersen method.

# Arguments
- `prob`: A `PricingProblem` containing an American `VanillaOption`.
- `method`: An `Andersen` pricing method.

# Returns
- An `AndersenSolution` containing price and stopping strategy.
"""
function solve(
    prob::PricingProblem{VanillaOption{TS,TE,American,C,S},I},
    method::A,
) where {TS,TE,I<:AbstractMarketInputs,C,S, A<:Andersen}

    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    sde_prob = sde_problem(prob, method.mc_method.dynamics, method.mc_method.strategy)
    sol = Hedgehog.simulate_paths(sde_prob, method.mc_method, method.mc_method.config.variance_reduction)
    spot_grid = Hedgehog.extract_spot_grid(sol)
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps+1, p])) for p = 1:npaths]

    # We need to begin by computing European option prices at every time step; how to do this efficiently? 
    estimated_H = Vector{eltype(spot_grid)}(undef, nsteps-1)
    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)

        isempty(in_the_money) && continue
        update_stopping_info!(estimated_H, stopping_info, method.policy, in_the_money, continuation, payoff_t, spot_grid[i, :], t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = mean(discounted_values)

    return AndersenSolution(prob, Andersen(method.mc_method, method.degree, estimate_andersen_policy(method.policy, estimated_H)), price, stopping_info, spot_grid)
end

function update_stopping_info!(
    estimated_H,
    stopping_info::Vector{Tuple{Int,U}},
    policy::AndersenPolicy1,
    paths::Vector{Int},
    cont_value::Vector{T},
    payoff_t::Vector{S},
    spot_grid,
    t::Int,
) where {T,S,U}
    display(policy)
    display(estimated_H)
    display(t)
    #value_from_H = h -> sum(payoff_t[p] for p in paths if payoff_t[p] > )
    error()
    exercise = payoff_t[paths] .> cont_value
    stopping_info[paths[exercise]] .= [(t, payoff_t[p]) for p in paths[exercise]]
end