############################################# 
# alm demo for a QP
#############################################

using ArgParse
using LineSearches
using Optim
using Random
using Plots
using Printf
using KrylovKit
using LaTeXStrings
using LinearAlgebra
using SparseArrays

# include("./hagerzhang.jl")
# include("./backtracking.jl")
# lsa::HagerZhangEx = HagerZhangEx(
# # linesearchmax=10
# )
# lsb::BackTrackingEx = BackTrackingEx(
#     ρ_hi=0.8,
#     ρ_lo=0.1,
#     order=2
# )
bool_plot = false
bool_opt = true
# bool_setup = true

function BacktrackLineSearch(
    f, g,
    gx, fx,
    x::Tx,
    s::Tg
) where {Tx,Tg}

    ϕ(α) = f(x .+ α .* s)
    function dϕ(α)

        gv = g(x + α .* s)

        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = f(x .+ α .* s)
        gv = g(x + α .* s)
        dphi = dot(gv, s)
        return (phi, dphi)
    end


    dϕ_0 = dot(s, gx)
    try
        α, fx = lsb(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        return α, fx
    catch y
        # throw(y)
        isa(y, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end

end
function HagerZhangLineSearch(
    f, g,
    gx, fx,
    x::Tx,
    s::Tg;
    α₀::R=1.0
) where {Tx,Tg,R<:Real}

    ϕ(α) = f(x .+ α .* s)
    function dϕ(α)

        gv = g(x + α .* s)

        return dot(gv, s)
    end
    function ϕdϕ(α)
        phi = f(x .+ α .* s)
        gv = g(x + α .* s)
        dphi = dot(gv, s)
        return (phi, dphi)
    end
    dϕ_0 = dot(s, gx)
    try
        α, fx, it = lsa(ϕ, dϕ, ϕdϕ, α₀, fx, dϕ_0)
        return α, fx, it
    catch u
        isa(u, LineSearchException) # && println() # todo
        return 0.1, fx, 1
    end

end
# tol 
ε = 1e-3
ϵₕ = 1e-7
opt_ls = :backtrack
if bool_setup
    Random.seed!(2)
    n = 150
    m = 100
    d = 20 # number constraints
    μ = 1e-1
    # linear pieces
    W = sprand(Float64, m, n, 0.3)
    w = rand(Float64, m) * 2 .- 1
    # constraints
    A = sprand(Float64, d, n, 0.2)
    b = A * rand(Float64, n)

    loss(x) = 0.5 * norm(W * x - w)^2 + 0.5 * μ * norm(x)^2
    grad(x) = W' * (W * x - w) + μ * x
    hess(x) = Symmetric(W' * W + μ * I)
    hvp(x, v) = W' * (W * v) + μ * v
    r(x) = A * x - b
    x₀ = ones(n) / 10
    y₀ = ones(d) / 10

end

r_hist, loss_hist = [], []
push!(r_hist, r(x₀))
push!(loss_hist, loss(x₀))

if bool_opt
    # Homogeneous ALM
    K = 25
    J = 10
    x = copy(x₀)
    y = copy(y₀)
    ρ = 1.0
    kₜ = 0
    for k in 1:K
        global x, y, ρ, ξ, d, θ, V, kₜ
        L(x) = loss(x) + y' * r(x) + ρ / 2 * (r(x)' * r(x))
        ϕ(x) = grad(x) + A' * (y + ρ * r(x))
        hvp₊(x, v) = hess(x) * v + ρ * A' * (A * v)
        δ = θ = 0
        β = 1e-3
        for j in 1:J
            # fix y, ρ, optimize
            _f = loss(x)
            _r = r(x)
            _p = _r' * _r
            _L = _f + y' * _r + ρ / 2 * _p
            _ϕ = ϕ(x)
            γ = 0.1

            @printf(
                "---- k:%3d, j:%3d, ρ:%.1e, |r|²:%.1e, f:%.2f, L₊:%.2f\n",
                k, j, ρ, _p, _f, _L
            )
            fail_count = 0
            while fail_count < 10 #true
                if (fail_count == 0) || (θ >= 0)
                    δ = _p * ρ * γ - (_f + y' * _r) * β * (1 - γ)
                else
                    if _p > ε # still large primal infeasibility
                        γ /= 20
                    else
                        denom = (1 + norm(d)^2)
                        ∂θ = -1 / denom
                        δ += θ * denom
                    end
                end
                if _ϕ |> norm < ε
                    break
                end

                fvp(v) = [
                    hvp₊(x, v[1:end-1]) + ϕ(x) * v[end];
                    ϕ(x)' * v[1:end-1] + δ * v[end]
                ]

                # compute eigenvalue from KrylovKit
                D, V, info = KrylovKit.eigsolve(
                    fvp, n + 1, 1, :SR, Float64;
                    tol=ϵₕ,
                    issymmetric=true, eager=true
                )
                ξ = V[1]
                d = ξ[1:end-1] ./ ξ[end]
                θ = -D[1]
                @printf(
                    "   |---- |d|:%.1e, |ϕ|:%.1e, θ:%.1e, δ:%+.1e \n",
                    d |> norm, _ϕ |> norm, θ, δ
                )
                kₜ += 1
                if θ < -1e-2
                    fail_count += 1
                    continue
                end
                break
            end

            α = 10.0
            kₗ = 1
            if opt_ls == :backtrack
                # use backtrack line-search algorithm
                df = 1e5
                Kₗ = 20
                dₙ = d |> norm
                while kₗ <= Kₗ && α > 1e-4
                    fx = L(x + d * α)
                    if fx - _L <= -1e-1 * α * dₙ
                        break
                    end
                    α *= 0.5
                    kₗ += 1
                end
            end

            @printf(
                "   |---- kₜ:%4d, kₗ:%2d, α:%.1e,\n",
                kₜ, kₗ, α
            )
            x += α * d
            if α < 1e-4
                break
            end
        end
        push!(r_hist, r(x))
        push!(loss_hist, loss(x))
        # if (norm(r(x)) < ε) || (abs(L(x) - loss(x)) < 1e-3)
        if (norm(ϕ(x)) < ε) || (abs(L(x) - loss(x)) < 1e-3)
            @info "terminated by residual"
            break
        end
        y += ρ * r(x)
        ρ *= 2
    end
end

