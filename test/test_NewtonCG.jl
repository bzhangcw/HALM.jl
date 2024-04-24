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
    include("problems/prob_fastdecayqp.jl")
end


ε = 1e-4
ε_CG = 1e-5
opt_ls = backtrack()
opt_inner = NewtonCG()

r_hist, loss_hist = [], []
cgtimes_hist = []
push!(r_hist, r(x₀))
push!(loss_hist, loss(x₀))

if bool_opt
    # ALM with Newton-CG
    K = 10
    J = 50
    x = copy(x₀)
    y = copy(y₀)
    ρ = 1.0
    for k in 1:K
        global x, y, ρ, ξ, d, θ, V, kₜ
        L(x) = loss(x) + y' * r(x) + ρ / 2 * (r(x)' * r(x))
        ϕ(x) = grad(x) + A' * (y + ρ * r(x))
        hvp₊(x, v) = hess(x) * v + ρ * A' * (A * v)

        _f = loss(x)
        _r = r(x)
        _p = _r' * _r

        @printf(
                "---- k:%3d, ρ:%.1e, |r|²:%.1e, f:%.2f\n",
                k, ρ, _p, _f
        )
        
        cgtimes_k = Vector{Float64}()
        # Newton-CG loop: fix y, ρ, optimize
        for j in 1:J
            _f = loss(x)
            _r = r(x)
            _p = _r' * _r
            _L = _f + y' * _r + ρ / 2 * _p
            _ϕ = ϕ(x)

            @printf(
                "   |---- j:%3d, |r|²:%.1e, f:%.2f, L₊:%.2f",
                j, _p, _f, _L
            )            

            # Use capped CG to find the Newton step
            cgtime = @elapsed d = search_direction(opt_inner::NewtonCG, x, _ϕ, hvp₊; rtol=ε_CG)
            push!(cgtimes_k, cgtime)

            # Line search
            α, kₗ = linearsearch(opt_ls, L, x, d, _L)

            @printf(
                ", kₗ:%2d, α:%.1e,\n",
                kₗ, α
            )
            x += α * d
            if α < 1e-4
                break
            end
        end
        push!(r_hist, r(x))
        push!(loss_hist, loss(x))
        push!(cgtimes_hist, cgtimes_k)
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


# plot(mean.(cgtimes_hist), label="Newton-CG")