############################################################
#### Convex QP with fast decayed spectrum               ####
####                                                    ####
####      minimize    0.5 * xᵀQx + cᵀx                  ####
####      subject to  Ax = b                            ####
####                                                    ####
#### Q: n x n PD matrix, c: n-vector, A: d x n matrix   ####    
############################################################
using LinearAlgebra
using Random
using SparseArrays

Random.seed!(123)

# Set up the problem dimensions
T = Float64
n = 100
d = 20  # number of constraints

# Generate fast decaying eigenvalues
rk = 10  # number of large eigenvalues
ord = 5  # λmax order of magnitude
eigs1 = [10^i for i in range(ord, 1, length=rk)]
eigs2 = [1 / i for i in 1:(n-rk)]
eigs = vcat(eigs1, eigs2)

# Generate symmetric matrix Q with given eigenvalues eigs and random vector c
U = qr(randn(T, n, n)).Q
Q = Symmetric(U * Diagonal(eigs) * U')
c = randn(T, n)

# Generate linear constraints
A = sprand(T, d, n, 0.2)
b = A * randn(T, n)

# functions
loss(x) = 0.5 * x' * (Q * x) + c' * x
grad(x) = Q * x + c
hess(x) = Q
hvp(x, v) = Q * v
r(x) = A * x - b

# initial guess
x₀ = ones(T, n) / 10
y₀ = ones(T, d) / 10

# Set up Gurobi model
if bool_gurobi
    model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    set_attribute(model, "OutputFlag", 0)
    @variable(model, varx[1:n])
    @constraint(model, A * varx .== b)
    @objective(model, Min, 0.5 * varx' * Q * varx + c' * varx)
    optimize!(model)
end