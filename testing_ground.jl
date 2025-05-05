using Revise, Hedgehog, BenchmarkTools, Dates, Random, Accessors, Polynomials

function fast_update_stopping_info!(    
    stopping_idx::Vector{Int},
    stopping_val::Vector{U},
    paths::Vector{Int},
    cont_value::Vector{T},
    payoff_t::Vector{S},
    t::Int,
) where {T,S,U}
    for (i, p) in enumerate(paths)
        payoff_t[p] > cont_value[i] || continue
        stopping_idx[p] = t
        stopping_val[p] = payoff_t[p]
    end
end

function solve_first(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
    sde_prob = Hedgehog.sde_problem(prob, method.mc_method.dynamics, method.mc_method.strategy)
    sol = Hedgehog.simulate_paths(sde_prob, method.mc_method, method.mc_method.config.variance_reduction)
    spot_grid = Hedgehog.extract_spot_grid(sol)
    return (T, sde_prob, sol, spot_grid)
end

function solve_orig(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
    first_part
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T, sde_prob, sol, spot_grid = first_part
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps+1, p])) for p = 1:npaths]

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        Hedgehog.update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = mean(discounted_values)

    return Hedgehog.LSMSolution(prob, method, price, stopping_info, spot_grid)
end

function solve_efficient_info(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
    first_part
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T, sde_prob, sol, spot_grid = first_part
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_idx = fill(nsteps, npaths)
    stopping_val = prob.payoff(view(spot_grid, nsteps+1, :))

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_idx[p] - t) * stopping_val[p] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        fast_update_stopping_info!(stopping_idx, stopping_val, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in zip(stopping_idx, stopping_val)]
    price = mean(discounted_values)

    stopping_info = collect(zip(stopping_idx, stopping_val))
    return Hedgehog.LSMSolution(prob, method, price, stopping_info, spot_grid)
end

function solve_efficient_info2(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
    first_part
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T, sde_prob, sol, spot_grid = first_part

    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_idx = fill(nsteps, npaths)
    stopping_val = prob.payoff(view(spot_grid, nsteps+1, :))

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            discount .^ (stopping_idx .- t) .* stopping_val

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        fast_update_stopping_info!(stopping_idx, stopping_val, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in zip(stopping_idx, stopping_val)]
    price = mean(discounted_values)

    stopping_info = collect(zip(stopping_idx, stopping_val))
    return Hedgehog.LSMSolution(prob, method, price, stopping_info, spot_grid)
end

function solve_quick_mean(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
    first_part
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T, sde_prob, sol, spot_grid = first_part
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps+1, p])) for p = 1:npaths]

    for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        Hedgehog.update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    price = mean(discount^t * val for (t, val) in stopping_info)

    return Hedgehog.LSMSolution(prob, method, price, stopping_info, spot_grid)
end

function solve_views(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
    first_part
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T, sde_prob, sol, spot_grid = first_part
    
    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid[nsteps+1, p])) for p = 1:npaths]

    @views for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid[i, :])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid[i, in_the_money]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        Hedgehog.update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = mean(discounted_values)

    return Hedgehog.LSMSolution(prob, method, price, stopping_info, spot_grid)
end

function solve_views_transpose(
    prob::Hedgehog.PricingProblem{Hedgehog.VanillaOption{TS,TE,Hedgehog.American,C,S},I},
    method::L,
    first_part
) where {TS,TE,I<:Hedgehog.AbstractMarketInputs,C,S, L<:Hedgehog.LSM}
    T, sde_prob, sol, spot_grid = first_part
    spot_grid_tr = permutedims(spot_grid)

    ntimes, npaths = size(spot_grid)
    nsteps = ntimes - 1
    discount = df(prob.market_inputs.rate, add_yearfrac(prob.market_inputs.referenceDate, T / nsteps))

    stopping_info = [(nsteps, prob.payoff(spot_grid_tr[p, nsteps+1])) for p = 1:npaths]

    @views for i = nsteps:-1:2
        t = i - 1

        continuation =
            [discount^(stopping_info[p][1] - t) * stopping_info[p][2] for p = 1:npaths]

        payoff_t = prob.payoff.(spot_grid_tr[:, i])
        in_the_money = findall(payoff_t .> 0)
        isempty(in_the_money) && continue

        x = spot_grid_tr[in_the_money, i]
        y = continuation[in_the_money]
        poly = Polynomials.fit(x, y, method.degree)
        cont_value = map(poly, x)

        Hedgehog.update_stopping_info!(stopping_info, in_the_money, cont_value, payoff_t, t)
    end

    discounted_values = [discount^t * val for (t, val) in stopping_info]
    price = mean(discounted_values)

    return Hedgehog.LSMSolution(prob, method, price, stopping_info, spot_grid)
end

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
    first_part = solve_first(prob, lsm_method)
    @time display(solve_orig(prob, lsm_method, first_part).price)
    @time display(solve_efficient_info(prob, lsm_method, first_part).price)
    @time display(solve_efficient_info2(prob, lsm_method, first_part).price)
    @time display(solve_quick_mean(prob, lsm_method, first_part).price)
    @time display(solve_views(prob, lsm_method, first_part).price)
    @time display(solve_views_transpose(prob, lsm_method, first_part).price)

    display(@benchmark solve_orig($prob, $lsm_method, $first_part))
    display(@benchmark solve_efficient_info($prob, $lsm_method, $first_part))
    display(@benchmark solve_efficient_info2($prob, $lsm_method, $first_part))
    display(@benchmark solve_quick_mean($prob, $lsm_method, $first_part))
    display(@benchmark solve_views($prob, $lsm_method, $first_part))
    display(@benchmark solve_views_transpose($prob, $lsm_method, $first_part))
    #display(@benchmark Hedgehog.solve($prob, $lsm_method))
    #display(@benchmark solve2($prob, $lsm_method))
end

test()