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

# include(srcdir("linearsearch.jl"))
# include(srcdir("innersolve.jl"))

bool_setup = true
bool_gurobi = false

if bool_setup
    # Set up the problem: define loss(), grad(), hess(), hvp(), r(), x₀, y₀
    include("problems/prob_regls.jl")
end

ATA = A' * A

δs = collect(range(0,-500, length=100))
θs = []
ts = []
vs = []

ϵₕ = 1e-7
x = copy(x₀)
y = copy(y₀)

ρ = 1.0

for δ in δs

    L(x) = loss(x) + y' * r(x) + ρ / 2 * (r(x)' * r(x))
    ϕ(x) = grad(x) + A' * (y + ρ * r(x))
    hvp₊(x, v) = hess(x) * v + ρ * A' * (A * v)

    # hessL(x) = hess(x) + ρ * ATA
    # _ϕ = ϕ(x)
    # F(x) = [hessL(x) _ϕ; _ϕ' zero(Float64)]

    fvp(v) = [
                    hvp₊(x, v[1:end-1]) + ϕ(x) * v[end];
                    ϕ(x)' * v[1:end-1] + δ * v[end]
                ]
    D, V, info = KrylovKit.eigsolve(
        fvp, n + 1, 1, :SR, Float64;
        tol=ϵₕ,
        issymmetric=true, eager=true
    )
    ξ = V[1]
    d = ξ[1:end-1] ./ ξ[end]
    θ = -D[1]

    push!(θs, θ)
    push!(ts, ξ[end])
    push!(vs, ξ[1:end-1])
    # eig_F = eigen(Matrix(F(x)))
    # cond_F = (eig_F.values[end] - eig_F.values[end-1]) / (eig_F.values[end] - eig_F.values[1])
    # push!(condnums_F, cond_F)
end

plot(
    plot(δs, θs, label=L"θ", xlabel="δ", ylabel=L"θ", title="Dual variable", lw=2, yscale=:log10),
    scatter(δs, ts, label="t", xlabel=L"δ", ylabel="t", title="t", lw=2)
)


