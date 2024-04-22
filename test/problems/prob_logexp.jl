########################################################################################
#### Non-convex log-exp example                                                     ####
####                                                                                ####
####      minimize    log(exp(Σ xᵢxᵢ₊₁ - 1) + 1) + log(exp(1 - Σ xᵢxᵢ₊₁) + 1)       ####
####      subject to  Ax = b                                                        ####
####                                                                                ####
########################################################################################
using Gurobi
using JuMP
using LinearAlgebra
using Random
using SparseArrays
const GRB_ENV = Gurobi.Env()

Random.seed!(123)

# Set up the problem dimensions
n = 100
d = 20  # number of constraints

# constraints
A = sprand(Float64, d, n, 0.2)
b = A * rand(Float64, n)


# 1-D helper functions for log(exp(t)+1)
h(t) = log(exp(t) + 1)
grad_h(t) = 1 - 1 / (exp(t) + 1)
hess_h(t) = exp(t) / (exp(t) + 1)^2
W = Tridiagonal(ones(n-1), zeros(n), ones(n-1))

# functions
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

# initial guess
x₀ = ones(n) / 10
y₀ = ones(d) / 10

nothing
