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
bool_setup = true
bool_plot = true
bool_opt = false
bool_opt_plot = false

# tol 
ε = 1e-3
ϵₕ = 1e-7
if bool_setup
    Random.seed!(2)
    n = 200
    m = 14
    d = 30 # number constraints
    μ = 1e-1
    # linear pieces
    W = sprand(Float64, m, n, 0.3) * 30
    w = (rand(Float64, m) * 2 .- 1) * 10
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
    K = 16
    kappa = zeros(K)
    kappaₗ = zeros(K)
    kappa₂ = zeros(K)
    ℓ₁ = zeros(K)
    ℓ₂ = zeros(K)

    J1 = zeros(K)
    J2 = zeros(K)
    J3 = zeros(K)
    Jₗ = zeros(K)
    R = zeros(K)
    ϵ = 1e-3
    ρ = 1.0
    # Homogeneous ALM
    x = copy(x₀)
    y = copy(y₀)
    kₜ = 0
end


for k in 1:K
    R[k] = ρ
    global x, y, ρ, ξ, d, θ, V, kₜ
    L(x) = loss(x) + y' * r(x) + ρ / 2 * (r(x)' * r(x))
    ϕ(x) = grad(x) + A' * (y + ρ * r(x))
    ℓ(x) = grad(x) + A' * y # without penalty term
    hvp₊(x, v) = hess(x) * v + ρ * A' * (A * v)
    δ = -ϵ


    H = Matrix(hess(x)) + ρ * A' * A + ϵ * I

    F₁ = Matrix([Matrix(hess(x)) ℓ(x); ℓ(x)' -ϵ])
    # F₂ = Matrix([A'A A'r(x); r(x)'A 0.1])
    F₂ = Matrix([A'A A'r(x); r(x)'A 5])
    F = F₁ + ρ * F₂

    D₁ = eigvals(F₁)
    D₂ = eigvals(F₂)
    # compute eigen-gap
    Δ₁ = D₁ .- D₁[1]
    Δ₁ = Δ₁[Δ₁.>1e-3]
    Δ₂ = D₂ .- D₂[1]
    Δ₂ = D₂[D₂.>1e-3]
    fvp(v) = F * v
    hv(v) = hvp₊(x, v) + ϵ * v

    D, V = eigen(H)
    Γ, P = eigen(F)
    Γ = Γ .- Γ[1]
    Γ₊ = Γ[Γ.>1e-3]

    @info "" Δ₁[end] Δ₁[1]
    @info "" Δ₂[end] Δ₂[1]
    @info "" Γ₊[end] Γ₊[1] ρ * Δ₂[1]
    @info "" Γ₊
    # @info D₁[end] / (D₁[2] - D₁[1])
    # record κ(H) and κ_L(F)
    # ℓ₁ = min(D₁[1], ρ * D₂[1])
    kappa[k] = cond(H)
    kappaₗ[k] = (Γ₊[end]) / (Γ₊[1])
    # kappa₂[k] = (ρ * Δ₂[end] + Δ₁[end]) / (ℓ₂)
    # kappa₂[k] = min(
    #     (ρ * Δ₂[end] + Δ₁[end]) / (ρ * Δ₂[1] + max(D₁[1], 0)),
    #     # Δ₁[end] / (Δ₁[1] + ρ * D₂[1]) + Δ₂[end] / (Δ₁[1] + ρ * D₂[1])
    #     # 1e5,
    #     (ρ * Δ₂[end] + Δ₁[end]) / (Δ₁[1] + max(ρ * D₂[1], 0))
    # )
    ℓ₁[k] = Δ₁[end] / Δ₁[1]
    ℓ₂[k] = Δ₂[end] / Δ₂[1]
    # estimate of condition number.
    # note,
    # F = [H ]

    # if compute the krylov steps
    # and record the krylov iterations.
    if bool_opt
        d, info1 = KrylovKit.linsolve(
            hv, -ϕ(x), -ϕ(x),
            CG(;
                tol=min(ϕ(x) |> norm, 1e-4),
                maxiter=n * 2,
                verbosity=3
            ),
        )

        d, info2 = KrylovKit.linsolve(
            hv, -ϕ(x), -ϕ(x),
            GMRES(;
                tol=min(ϕ(x) |> norm, 1e-4),
                maxiter=n * 2,
                krylovdim=n * 2,
                verbosity=3
            ),
        )
        # d, info3 = KrylovKit.linsolve(
        #     hv, -ϕ(x), -ϕ(x),
        #     GMRES(;
        #         tol=min(ϕ(x) |> norm, 1e-4),
        #         maxiter=n * 2,
        #         verbosity=3
        #     ),
        # )

        Λ, U, infoₗ = KrylovKit.eigsolve(
            fvp, rand(n + 1), 1, :SR;
            tol=1e-8,
            issymmetric=true, eager=true, verbosity=3
        )
        J1[k] = info1.numops
        J2[k] = info2.numops
        # J3[k] = info3.numops
        Jₗ[k] = infoₗ.numops
        ##############################
    end
    ρ *= 2
end


pgfplotsx()
# f2 = plot(
#     size=(600, 650),
#     # size=(1400, 1000),
#     # yscale=:log10,
#     # yticks=[1e-2, 1e-1, 2e-1, 5e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 1e4, 1e8, 1e12],
#     labelfontsize=26,
#     xscale=:log2,
#     xtickfont=font(26),
#     ytickfont=font(26),
#     legendfontsize=26,
#     titlefontsize=26,
#     leg=:topright,
#     legendfonthalign=:left,
#     legendfontfamily="sans-serif",
#     xaxis=L"\rho",
# )
# plot!(
#     R, kappaₗ,
#     label=L"\kappa_L(F_+)",
#     linewidth=4
# )

# plot!(
#     R, kappa₂,
#     label=L"\kappa_L(F_+) \quad\textrm{estimate}",
#     linewidth=4
# )

# display(f2)

pgfplotsx()
f4 = plot(
    size=(600, 650),
    # size=(1400, 1000),
    yscale=:log10,
    xscale=:log2,
    yticks=[1e-2, 1e-1, 2e-1, 5e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e8, 1e12],
    xticks=[2, 4, 32, 64, 128, 256, 512, 1024, 2048, 1048576, 2^K],
    labelfontsize=26,
    xtickfont=font(26),
    ytickfont=font(26),
    legendfontsize=26,
    titlefontsize=26,
    leg=:topleft,
    legendfonthalign=:left,
    legendfontfamily="sans-serif",
    xaxis=L"\rho",
)

plot!(
    R, kappa,
    label=L"\kappa(\nabla^2 L_+)",
    linewidth=4
)

plot!(
    R, kappaₗ,
    label=L"\kappa_L(F_+)",
    linewidth=4
)

# plot!(
#     R, kappa₂,
#     label=L"\kappa_L(F_+)~\textrm{est.}",
#     linewidth=4,
#     linestyle=:dash
# )
plot!(
    R, ℓ₁,
    linewidth=2,
    linestyle=:dash,
    label=:none
)
plot!(
    R, ℓ₂,
    linewidth=2,
    linestyle=:dash,
    label=:none
)

display(f4)

if bool_opt_plot
    f3 = plot(
        size=(600, 650),
        # size=(1400, 1000),
        # yscale=:log10,
        xscale=:log2,
        # yticks=[1e-2, 1e-1, 2e-1, 5e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e8, 1e12],
        # xticks=[2, 4, 32, 64, 128, 256, 512, 1024, 2048, 1048576, 2^K],
        labelfontsize=26,
        xtickfont=font(26),
        ytickfont=font(26),
        legendfontsize=26,
        titlefontsize=26,
        leg=:topleft,
        legendfonthalign=:left,
        legendfontfamily="sans-serif",
        lagendalpha=0.8,
        xaxis=L"\rho",
    )

    plot!(
        R, J1,
        label=L"\texttt{CG}",
        linewidth=4
    )
    plot!(
        R, J2,
        label=L"\texttt{GMRES}",
        linewidth=4
    )
    # plot!(
    #     R, J3,
    #     label=L"\texttt{rGMRES}",
    #     linewidth=4
    # )
    plot!(
        R, Jₗ,
        label=L"\texttt{Lanczos}",
        linewidth=4
    )
    display(f3)
end