using DrWatson
@quickactivate "."
using Gurobi
using JuMP
using KrylovKit
using LaTeXStrings
using LinearAlgebra
using Plots
using Printf
using Statistics

include(srcdir("linearsearch.jl"))
include(srcdir("innersolve.jl"))

bool_setup = true
bool_opt = true
bool_gurobi = true
bool_plot = false

if bool_setup
    # Set up the problem: define loss(), grad(), hess(), hvp(), r(), x₀, y₀
    probname = "regls"
    include("problems/prob_" * probname * ".jl")
end


ε = 1e-4
ϵₕ = 1e-8
opt_ls = backtrack()


r_hist, loss_hist = [], []
lanczostimes_hist = []
cgtimes_hist = []
push!(r_hist, r(x₀))
push!(loss_hist, loss(x₀))

condnums_hessL = []
condnums_F = []
ρs = []

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
        hvp₊(x, v) = hvp(x, v) + ρ * A' * (A * v)
        δ = 0
        lanczostimes_k = []
        cgtimes_k = []
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
            δᵤ = _p * ρ

            while failed_count < 10

                if _p / norm(_ϕ) >= 1e-1
                    # large primal infeasibility 
                    δ = _p * ρ * γ - (_f + y' * _r) * (1 - γ)
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
                lanczos_time = @elapsed begin
                    D, V, info = KrylovKit.eigsolve(
                        fvp, n + 1, 1, :SR, Float64;
                        tol=ϵₕ,
                        issymmetric=true, eager=true
                    )
                    ξ = V[1]
                    d = ξ[1:end-1] ./ ξ[end]
                    θ = -D[1]
                end
                @printf(
                    "   |---- |d|:%.1e, |ϕ|:%.1e, θ:%.1e, δ:%+.1e \n",
                    d |> norm, _ϕ |> norm, θ, δ
                )
                kₜ += 1
                if θ < 0
                    γ /= 20
                    failed_count += 1
                    continue
                end
                # Keep the Lanczos time accepted θ
                push!(lanczostimes_k, lanczos_time)
                break
            end

            # if bool_plot
            #     cgtime = @elapsed begin
            #         d, _ = KrylovKit.linsolve(
            #             (v -> hvp₊(x, v)), -ϕ(x), zeros(n), rtol=1e-5, issymmetric=true
            #         )
            #     end
            #     push!(cgtimes_k, cgtime)

            #     hessL(x) = hess(x) + ρ * A' * A
            #     F(x) = [hessL(x) _ϕ; _ϕ' δ]
            #     push!(ρs, ρ)
            #     push!(condnums_hessL, cond(Matrix(hessL(x)), 2))
            #     eig_F = eigen(Matrix(F(x)))
            #     cond_F = (eig_F.values[end] - eig_F.values[end-1]) / (eig_F.values[end] - eig_F.values[1])
            #     push!(condnums_F, cond_F)
            # end

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
        push!(lanczostimes_hist, lanczostimes_k)
        push!(cgtimes_hist, cgtimes_k)
        if (norm(r(x)) < ε) || (abs(L(x) - loss(x)) < 1e-3)
            @info "terminated by residual"
            break
        end
        y += ρ * r(x)
        (ρ *= 2)
    end
end

# Compare with Gurobi
if bool_gurobi
    println("Gurobi: optimal value = ", objective_value(model))
    @printf("Difference of optimal values = %.2e\n", objective_value(model) - loss(x) |> abs)
end

if bool_plot
    plot(ρs, condnums_hessL, label=L"κ(∇^2L)", xlabel="ρ", ylabel="condition number", title="Condition numbers", lw=2)
    plot!(ρs, condnums_F, label=L"κ(F)", lw=2)
    savefig("figs/" * probname * "_condnum.png")

    plot(mean.(lanczostimes_hist), label="Lanczos time", xlabel="iter", ylabel="time (s)", title="Time", lw=2)
    plot!(mean.(cgtimes_hist), label="CG time", lw=2)
    savefig("figs/" * probname * "_time.png")
end