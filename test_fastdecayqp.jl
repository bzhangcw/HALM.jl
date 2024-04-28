using LinearAlgebra
using Random
using JuMP, Gurobi
const GRB_ENV = Gurobi.Env()

Random.seed!(123)

n = 100
d = 20  # number of constraints
rk = 10  # number of fast decaying eigenvalues
eigs1 = [10^i for i in range(5, 1, length=rk)]
eigs2 = [1 / i for i in 1:(n-rk)]
eigs = vcat(eigs1, eigs2)

# Generate symmetric matrix Q
U = qr(randn(n, n)).Q
Q = Symmetric(U * Diagonal(eigs) * U')

# Generate random vector c
c = randn(n)

# Generate linear constraints
A = randn(d, n)
b = A * randn(n)

# Set up functions and initial values
loss(x) = 0.5 * x' * (Q * x) + c' * x
grad(x) = Q * x + c
hess(x) = Q
hvp(x, v) = Q * v
r(x) = A * x - b
x₀ = ones(n) / 10
y₀ = ones(d) / 10

bool_setup = false
include("test_toyqp.jl")

# Solved by Gurobi
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(model, "OutputFlag", 0)
@variable(model, var[1:n])
@constraint(model, A * var .== b)
@objective(model, Min, 0.5 * var' * Q * var + c' * var)
optimize!(model)
println("Gurobi: optimal value = ", objective_value(model))