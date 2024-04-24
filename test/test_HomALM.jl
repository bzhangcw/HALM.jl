using DrWatson
@quickactivate "."
using Gurobi
using JuMP
using KrylovKit
using LinearAlgebra
using Plots
using Printf
using Statistics

include(srcdir("linearsearch.jl"))
include(srcdir("innersolve.jl"))

bool_setup = true
bool_opt = true
bool_gurobi = true

if bool_setup
    # Set up the problem
    include("problems/prob_regls.jl")
end


ε = 1e-4
ϵₕ = 1e-7
opt_ls = backtrack()


r_hist, loss_hist = [], []
cgtimes_hist = []
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
        δ = 0
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
            failed_count = 0
            while failed_count < 10
                δ = _p * ρ * γ - _L * 1e-3 * (1 - γ)
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
                if θ < 0
                    γ /= 20
                    fail_count += 1
                    continue
                end
                break
            end

            # Line search
            α, kₗ = linearsearch(opt_ls, L, x, d, _L)

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
        if (norm(r(x)) < ε) || (abs(L(x) - loss(x)) < 1e-3)
            @info "terminated by residual"
            break
        end
        y += ρ * r(x)
        ρ *= 2
    end
end

# Compare with Gurobi
if bool_gurobi
    println("Gurobi: optimal value = ", objective_value(model))
    @printf("Difference of optimal values = %.2e\n", objective_value(model) - loss(x) |> abs)
end