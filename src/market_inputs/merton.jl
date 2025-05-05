using JumpProcesses
"""
    MertonInputs <: AbstractMarketInputs

Market data inputs for the Merton model, which extends the Black-Scholes model with Poisson-arriving LogNormal jumps. 

# Fields
- `referenceDate`: The date from which maturity is measured.
- `rate`: The risk-free interest rate (annualized).
- `spot`: The current spot price of the underlying asset.
- `sigma`: The volatility of the underlying asset (annualized).
- `lambda`: The intensity of the Poisson process (average number of jumps per unit time).
- `mu`: The average jump size (log-normal mean).
- `sigma_jump`: The standard deviation of the jump size (log-normal standard deviation).

This struct encapsulates the necessary inputs for pricing derivatives under the Merton model.
"""
struct MertonInputs{R <: AbstractRateCurve,TRef <: Real, TSpot <: Real, TSigma <: AbstractVolSurface, TLambda<:Real, TMuJump<:Real, TSigmaJump<:Real} <: AbstractMarketInputs
    referenceDate::TRef
    rate::R
    spot::TSpot
    sigma::TSigma
    lambda::TLambda
    mu_jump::TMuJump
    sigma_jump::TSigmaJump
end

MertonInputs(reference_date::TimeType, rate::AbstractRateCurve, spot, sigma::Real, lambda, mu_jump, sigma_jump) =
    MertonInputs(to_ticks(reference_date), rate, spot, FlatVolSurface(sigma; reference_date=to_ticks(reference_date)), lambda, mu_jump, sigma_jump)

MertonInputs(reference_date::TimeType, rate::Real, spot, sigma, lambda, mu_jump, sigma_jump) = MertonInputs(
    reference_date,
    FlatRateCurve(rate; reference_date = reference_date),
    spot,
    sigma,
    lambda,
    mu_jump,
    sigma_jump
)

function LogGBMJProblem(μ, σ, λ, μⱼ, σⱼ, u0, tspan; seed = UInt64(0), kwargs...)
    f = function (u, p, t)
        return @. μ - 0.5 * σ^2  # Drift of log(S_t)
    end
    g = function (u, p, t)
        return @. σ  # Constant diffusion for log(S_t)
    end

    J = LogNormal(μⱼ, σⱼ)

    affect! = function (integrator)
        @. integrator.u[1] *= Distributions.rand(J)
        nothing
    end

    const_jump = ConstantRateJump(λ, affect!)

    noise = WienerProcess(tspan[1], 0.0)

    sde_f = SDEFunction(f, g)
    sde_prob = SDEProblem{false}(
        sde_f,
        u0,
        (tspan[1], tspan[2]),
        noise = noise,
        seed = seed,
        kwargs...,
    )
    return JumpProblem(sde_prob, Direct(), const_jump)
end

function LogGBMJProblem!(μ, σ, u0, tspan; seed = UInt64(0), kwargs...)
    f! = function (du, u, p, t)
        @. du = μ - 0.5 * σ^2  # Drift of log(S_t)
    end
    g! = function (du, u, p, t)
        @. du = σ  # Constant diffusion
    end

    J = LogNormal(μⱼ, σⱼ) # LogNormal jumps
    affect! = function (integrator)
        @. integrator.u[1] *= Distributions.rand(J)
        nothing
    end

    const_jump = ConstantRateJump(λ, affect!)

    noise = WienerProcess(tspan[1], 0.0)

    sde_f = SDEFunction{true}(f!, g!)
    sde_prob = return SDEProblem(
        sde_f,
        u0,
        (tspan[1], tspan[2]),
        noise = noise,
        seed = seed,
        kwargs...,
    )
    return JumpProblem(sde_prob, Direct(), const_jump)
end