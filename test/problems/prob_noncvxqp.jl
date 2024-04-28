####################################################################
#### Non-convex QP                                              ####
####                                                            ####
####      minimize    0.5 * xᵀQx + cᵀx                          ####
####      subject to  Ax = b                                    ####
####                                                            ####
#### Q: n x n indefinite matrix, c: n-vector, A: d x n matrix   ####    
####################################################################
using Gurobi
using JuMP
using LinearAlgebra
using Random
using SparseArrays

Random.seed!(123)

# Set up the problem dimensions
T = Float64
n = 50
d = 10  # number of constraints

# Generate an orthogonal matrix [ V | U ]
VU = qr(randn(T, n, n)).Q
V = @views VU[:, 1:(n-d)]
U = @views VU[:, (n-d+1):n]

# Take A as Uᵀ and generate b
A = U'
x₀ = U * randn(T, d)
b = A * x₀

# Generate Q
D₁ = Diagonal(rand(T, n-d)) # positive diagonal
D₂ = Diagonal(randn(T, d))  # can have negative diagonal
D = [D₁ spzeros(T, n-d, d); spzeros(T, d, n-d) D₂]
Q = VU * D * VU'

# Generate vector c such that Q x₀ + c ∈ range(VᵀQV)
c = Q * (V * randn(T, n-d) - x₀)

# functions
loss(x) = 0.5 * x' * (Q * x) + c' * x
grad(x) = Q * x + c
hess(x) = Q
hvp(x, v) = Q * v
r(x) = A * x - b

# initial guess
x₀ = ones(T, n) / 10
y₀ = ones(T, d) / 10

# Analytical optimal solution
u_opt = D₁ \ (-V' * (Q * x₀ + c))
fx₀ = 0.5 * x₀' * Q * x₀ + c' * x₀
opt = fx₀ + 0.5 * u_opt' * D₁ * u_opt + (Q * x₀ + c)' * (V * u_opt)
