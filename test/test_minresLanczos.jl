using LinearAlgebra
using Random

n = 100
m = 20  # number of constraints
r = 10  # number of fast decaying eigenvalues
d1 = [10^i for i in range(10, 1, length=r)]
d2 = [1 / i for i in 1:(n-r)]
d = vcat(d1, d2)

# Generate symmetric matrix Q
U = qr(randn(T, n, n)).Q
Q = Symmetric(U * Diagonal(d) * U')

# Generate random vector c
c = randn(n)

# Generate linear constraints
A = randn(m, n)
b = A * randn(n)