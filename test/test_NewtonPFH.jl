############################################################
# We use the same structure as in test_NewtonCG
#   but we substitute loop j by 
#   the Path-Following Homogeneous method (PFH) 
#   as the subproblem solver.
#   we can directly use DRSOM as a black
############################################################
using DrWatson
@quickactivate "."
using Gurobi
using JuMP
using KrylovKit
using LinearAlgebra, LaTeXStrings
using Plots
using Printf
using Statistics
using DRSOM

include(srcdir("linearsearch.jl"))
include(srcdir("innersolve.jl"))

bool_setup = true
bool_opt = true
bool_gurobi = true
bool_plot = true

if bool_setup
    # Set up the problem: define loss(), grad(), hess(), hvp(), r(), x₀, y₀
    probname = "regls"
    include("problems/prob_" * probname * ".jl")
end


ε = 1e-3
ε_CG = 1e-7
opt_ls = backtrack()
opt_inner = NewtonCG()

r_hist, loss_hist = [], []
cgtimes_hist = []

condnums_hessL = []
condnums_F = []
ρs = []

push!(r_hist, r(x₀))
push!(loss_hist, loss(x₀))

if bool_opt
    # ALM with Newton-CG
    K = 20
    J = 50
    x = copy(x₀)
    y = copy(y₀)
    ρ = 1.0
    Hv = similar(x)
    j = 0
    for k in 1:K
        global x, y, ρ, ξ, d, θ, V, kₜ, j
        L(x) = loss(x) + y' * r(x) + ρ / 2 * (r(x)' * r(x))
        ϕ(x) = grad(x) + A' * (y + ρ * r(x))
        # hvp₊(x, v) = hvp(x, v) + ρ * A' * (A * v)
        hvp₊(x, v, Hv) = copyto!(Hv, hvp(x, v) + ρ * A' * (A * v))

        _f = loss(x)
        _r = r(x)
        _p = _r' * _r

        @printf(
            "---- k:%3d, j:%3d, ρ:%.1e, |r|²:%.1e, f:%.2f\n",
            k, j, ρ, _p, _f
        )

        cgtimes_k = Vector{Float64}()
        # Newton-CG loop: fix y, ρ, optimize
        # todo, use PFHIteration instead later
        rh = PFH(name=Symbol("PF-HSODM"))(;
            x0=copy(x), f=L, g=ϕ, hvp=hvp₊,
            maxiter=10, tol=ε, freq=1,
            step=:hsodm, μ₀=5e-2,
            bool_trace=true,
            verbose=0,
            direction=:warm,
            maxtime=500,
        )

        if bool_plot
            hessL(x) = hess(x) + ρ * A' * A
            # homotopy matrix
            F(x) = [hessL(x) ϕ(x); ϕ(x)' 0]
            push!(ρs, ρ)
            push!(condnums_hessL, cond(Matrix(hessL(x)), 2))
            eig_F = eigen(Matrix(F(x)))
            cond_F = (eig_F.values[end] - eig_F.values[end-1]) / (eig_F.values[end] - eig_F.values[1])
            push!(condnums_F, cond_F)
        end

        j += rh.state.kᵤ
        x = rh.state.x
        push!(r_hist, r(x))
        push!(loss_hist, loss(x))
        push!(cgtimes_hist, cgtimes_k)
        if (norm(kkt(x, y)) < 1e-3)
            @info "terminated by residual"
            break
        end
        y += ρ * r(x)
        (norm(r(x)) > ε) && (ρ *= 2)
    end
end

# Compare with Gurobi
if bool_gurobi
    println("Gurobi: optimal value = ", objective_value(model))
    @printf("Difference of optimal values = %.2e\n", objective_value(model) - loss(x) |> abs)
    @printf("KKT values = %.2e: %.2e\n", kkt(value.(varx), -dual.(consy)) |> norm, kkt(x, y) |> norm)
end

if bool_plot
    plot(ρs, condnums_hessL, label=L"κ(∇^2L)", xlabel="ρ", ylabel="condition number", title="Condition numbers", lw=2)
    plot!(ρs, condnums_F, label=L"κ(F)", lw=2)
    savefig("figs/" * probname * "_condnum.png")
end