############################################# 
# alm demo for a toy 2D nonconvex problem
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
using JuMP, Gurobi
const GRB_ENV = Gurobi.Env()

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
bool_setup = true

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
    d = 100 # number constraints

    # helper functions
    h(t) = log(exp(t) + 1)
    grad_h(t) = 1 - 1 / (exp(t) + 1)
    hess_h(t) = exp(t) / (exp(t) + 1)^2
    W = Tridiagonal(ones(n-1), zeros(n), ones(n-1))

    # constraints
    A = sprand(Float64, d, n, 0.2)
    b = A * rand(Float64, n)

    loss(x) = h(sum([x[i] * x[i+1] for i in 1:n-1]) - 1) + 
              h(1 - sum([x[i] * x[i+1] for i in 1:n-1]))
    function grad(x)
        xx = sum([x[i] * x[i+1] for i in 1:n-1])
        return (h(xx - 1) - h(1 - xx)) * W * x 
    end
    function hess(x)
        xx = sum([x[i] * x[i+1] for i in 1:n-1])
        c = grad_h(xx - 1) - grad_h(1 - xx)
        return Symmetric(c * W + W * x * x' * W)
    end
    function hvp(x, v)
        xx = sum([x[i] * x[i+1] for i in 1:n-1])
        c = grad_h(xx - 1) - grad_h(1 - xx)
        return c * (W * v) + (W * x) * (x' * (W * v))
    end
    r(x) = A * x - b
    x₀ = ones(n) / 10
    y₀ = ones(d) / 10

    r_hist, loss_hist = [], []
    push!(r_hist, r(x₀))
    push!(loss_hist, loss(x₀))
end
if bool_opt
    # Homogeneous ALM
    K = 10  # number of outer iterations (ALM iterations)
    J = 10  # number of inner iterations (for solving subproblem)
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
            κ = 0.1

            @printf(
                "---- k:%3d, j:%3d, ρ:%.1e, |r|²:%.1e, f:%.2f, L₊:%.2f\n",
                k, j, ρ, _p, _f, _L
            )
            while true
                # δ = _p * ρ * γ - _L * 1e-3 * (1 - γ)
                δ = -_p * ρ * γ - κ * sqrt(norm(grad(x) + A' * y))
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
        end # End of inner loop
        push!(r_hist, r(x))
        push!(loss_hist, loss(x))
        if (norm(r(x)) < ε) || (abs(L(x) - loss(x)) < 1e-3)
            @info "terminated by residual"
            break
        end
        y += ρ * r(x)
        # ρ *= 2
    end
end

r_last = @sprintf("%.3e", norm(r_hist[end]))
loss_last = @sprintf("%.4f", loss_hist[end])
plt = plot(
    plot(norm.(r_hist), label=L"\|Ax - b\|_2", title=L"$r_n =$"*r_last , yaxis=:log),
    plot(loss_hist, label = L"loss($x$)", title=L"$x_n =$"*loss_last),
    layout=(1, 2),
    size=(800, 400)
)
display(plt)
savefig(plt, "fig/cvx_logexp_double_fixedrho.png")


# Solved by Gurobi
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(model, "OutputFlag", 0)
@variable(model, var[1:n])
@constraint(model, A * var .== b)
@objective(model, Min, log(exp(sum([x[i] * x[i+1] for i in 1:n-1]) - 1) + 1) + log(exp(1 - sum([x[i] * x[i+1] for i in 1:n-1])) + 1))
optimize!(model)
println("Gurobi: optimal value = ", objective_value(model))