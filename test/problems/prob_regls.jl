############################################################
#### Regularized Least Squares problem                  ####
####                                                    ####
####      minimize 0.5 * |Wx - w|^2 + 0.5 * μ * |x|^2   ####
####      subject to Ax = b                             ####
####                                                    ####
#### W: m x n matrix, w: m-vector, A: d x n matrix,     ####    
############################################################
using Gurobi
using JuMP
using LinearAlgebra
using Random
using SparseArrays
const GRB_ENV = Gurobi.Env()

Random.seed!(123)

# Set up the problem dimensions and parameters
T = Float64
n = 1500
m = 600
d = 180      # number constraints
μ = 1e-6    # regularization parameter

# objective pieces
W = sprand(T, m, n, 0.3)
w = rand(T, m) * 2 .- 1
# constraints
A = sprand(T, d, n, 0.2)
b = A * rand(T, n)

# functions
loss(x) = 0.5 * norm(W * x - w)^2 + 0.5 * μ * norm(x)^2
grad(x) = W' * (W * x - w) + μ * x
hess(x) = Symmetric(W' * W + μ * I)
hvp(x, v) = W' * (W * v) + μ * v
r(x) = A * x - b
kkt(x, y) = [grad(x) + A' * y; A * x - b]

# initial guess
x₀ = ones(n) / 10
y₀ = ones(d) / 10

# Set up Gurobi model
if bool_gurobi
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_attribute(model, "OutputFlag", 1)
    @variable(model, varx[1:n])
    consy = @constraint(model, A * varx .== b)
    @objective(model, Min, 0.5 * (W * varx - w)' * (W * varx - w) + 0.5 * μ * varx' * varx)
    optimize!(model)
end