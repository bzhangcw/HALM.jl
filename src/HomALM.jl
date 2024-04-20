using LinearAlgebra
using Plots
# ENV["GUROBI_HOME"] = "/Library/gurobi1100/macos_universal2"
using JuMP, Gurobi
const GRB_ENV = Gurobi.Env()

"""
Structure for linearly constrained minimization problem

    min  f(x)
    s.t. Ax = b

- H is the Hessian of f(x) 
- g is the gradient of f(x).
"""
Base.@kwdef struct LinearConstrProblem
    f::Function
    H::Function     # Hessian of f(x)
    g::Function     # Gradient of f(x)
    A::Array{Float64, 2}
    b::Array{Float64, 1}
end

"""
Augmented Lagrangian function for LinearConstrProblem

    L(x, y) = f(x) + y' * (Ax - b) + 0.5 * ρ * ||Ax - b||²
"""
function Aug_Lagrangian(Prob::LinearConstrProblem, x, y, ρ) 
    return Prob.f(x) + y' * (Prob.A * x - Prob.b) + 0.5 * ρ * norm(Prob.A * x - Prob.b)^2
end


function solve_ALM_subproblem(subprob_model, fex, Prob::LinearConstrProblem, y, ρ)
    A = Prob.A
    b = Prob.b
    @objective(subprob_model, Min, fex + y' * (A * var_x - b) + 0.5 * ρ * (A * var_x - b)' * (A * var_x - b))    
    optimize!(subprob_model)    
    return value.(var_x)
end

function ALM(Prob::LinearConstrProblem, x0, y0, ρ, subprob_model, fex; tol = 1e-6, max_iter = 500)
    A = Prob.A
    b = Prob.b
    
    x = x0
    y = y0
    iter = 0
    
    while (iter < max_iter) && (norm(A * x - b) > tol)
        iter += 1

        # Update x
        x = solve_ALM_subproblem(subprob_model, fex, Prob, y, ρ)
        # Update y
        y = y + ρ * (A * x - b)
    end
    return x, iter
end

function HomALM(Prob::LinearConstrProblem, x0, y0, ρ; tol = 1e-6, max_iter = 500)
    f = Prob.f
    H = Prob.H
    g = Prob.g
    A = Prob.A
    b = Prob.b
    
    x = x0
    y = y0
    Δ = 2 * sqrt(tol)
    iter = 0
    x_history = [x]
    L_history = [Aug_Lagrangian(Prob, x, y, ρ)]
    
    while (iter < max_iter) && (norm(A * x - b) > tol)
        iter += 1

        # Update δ
        δ = update_δ(Prob, x, y, ρ)

        # Update x
        ϕ = g(x) + A' * y + ρ * A' * (A * x - b) 
        F = [0.5*H(x)-(ρ/2)*A'*A ϕ; ϕ' -δ]
        eigF = eigen(F)
        v = eigF.vectors[:, 1]
        
        if abs(v[end]) > 0.1
            d = v[1:end-1] / v[end]
        else
            d = v[1:end-1]
        end

        if norm(d) > Δ
            η = Δ / norm(d)
            x = x + η * d
        else
            x = x + d
        end
        
        # Update y
        y = y + ρ * (A * x - b)
        # x_old = x_history[end]
        # y = y - A * inv(H(x_old) + ρ * A' * A) * A' * (A * x_old - b) # second order update

        push!(x_history, x)
        push!(L_history, Aug_Lagrangian(Prob, x, y, ρ))
    end

    return x, iter, x_history, L_history
end

function update_δ(Prob::LinearConstrProblem, x, y, ρ; κ = 0.1, γ = 0.8)
    δf = κ * sqrt(norm(Prob.g(x) + Prob.A' * y))
    return γ * δf + (1 - γ) * ρ * norm(Prob.A * x - Prob.b)^2
end