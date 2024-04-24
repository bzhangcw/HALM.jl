using JuMP, Gurobi
const GRB_ENV = Gurobi.Env()

include("test_utils.jl")

"""
Consider the linearly constrained minimization problem
      min  f(x)
      s.t. Ax = b
Use augmented Lagrangian method and homogenized variant to solve the problem.
"""


"""
Example 1:

    minimize f(x) = x₁² + x₂² subject to x₁ + x₂ = 1.

The augmented Lagrangian function is

      L(x, y) = x₁² + x₂² + y(x₁ + x₂ - 1) + ρ/2 * ||x₁ + x₂ - 1||²

"""
f(x) = sum(x.^2)
H(x) = I
g(x) = [2*x[1]; 2*x[2]]
A = [1.0 1.0]
b = [1.0]
Prob = LinearConstrProblem(f, H, g, A, b)

# Define JuMP ALM subproblem model with expression for f(x)
subprob_model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(subprob_model, "OutputFlag", 0)
@variable(subprob_model, var_x[1:2])
fex = @expression(subprob_model, var_x[1]^2 + var_x[2]^2) # Expression for f(x)

# Define JuMP model for original problem (to be solved by Gurobi)
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(model, "OutputFlag", 0)
@variable(model, var[1:2])
@constraint(model, A * var .== b)
@objective(model, Min, var[1]^2 + var[2]^2)

## Initial point and penalty parameter for ALM
x0 = [0.0, 0.0]
y0 = [0.0]
ρ = 1.0
gif = test_fun(Prob, x0, y0, ρ, model, subprob_model, fex; run_ALM=false)


"""
Example 2:

    minimize f(x) = 2x₁² + x₂² subject to x₁ + x₂ = 1.

The augmented Lagrangian function is

      L(x, y) = 2x₁² + x₂² + y(x₁ + x₂ - 1) + ρ/2 * ||x₁ + x₂ - 1||²

"""
f(x) = x[1]^2 + 2 * x[2]^2
H(x) = [1.0 0.0; 0.0 2.0]
g(x) = [2*x[1]; 4*x[2]]
A = [1.0 1.0]
b = [1.0]
Prob = LinearConstrProblem(f, H, g, A, b)

# Define JuMP ALM subproblem model with expression for f(x)
subprob_model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(subprob_model, "OutputFlag", 0)
@variable(subprob_model, var_x[1:2])
fex = @expression(subprob_model, var_x[1]^2 + 2 * var_x[2]^2) # Expression for f(x)

# Define JuMP model for original problem (to be solved by Gurobi)
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(model, "OutputFlag", 0)
@variable(model, var[1:2])
@constraint(model, A * var .== b)
@objective(model, Min, var[1]^2 + 2 * var[2]^2)

## Initial point and penalty parameter for ALM
x0 = [0.0, 0.0]
y0 = [0.0]
ρ = 1.0
gif = test_fun(Prob, x0, y0, ρ, model, subprob_model, fex; run_ALM=false)


"""
Example 3:

    minimize f(x) = -x₁² + x₂² subject to x₁ + x₂ = 1.

The augmented Lagrangian function is

      L(x, y) = -x₁² + x₂² + y(x₁ + x₂ - 1) + ρ/2 * ||x₁ + x₂ - 1||²

"""
f(x) = -x[1]^2 + 2 * x[2]^2
H(x) = [-1.0 0.0; 0.0 2.0]
g(x) = [-2*x[1]; 4*x[2]]
A = [1.0 1.0]
b = [1.0]
Prob = LinearConstrProblem(f, H, g, A, b)

# Define JuMP ALM subproblem model with expression for f(x)
subprob_model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(subprob_model, "OutputFlag", 0)
@variable(subprob_model, var_x[1:2])
fex = @expression(subprob_model, -var_x[1]^2 + 2 * var_x[2]^2) # Expression for f(x)

# Define JuMP model for original problem (to be solved by Gurobi)
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(model, "OutputFlag", 0)
@variable(model, var[1:2])
@constraint(model, A * var .== b)
@objective(model, Min, -var[1]^2 + 2 * var[2]^2)

## Initial point and penalty parameter for ALM
x0 = [0.0, 0.0]
y0 = [0.0]
ρ = 1.0
gif = test_fun(Prob, x0, y0, ρ, model, subprob_model, fex; run_ALM=false)


"""
Example 4:

    minimize      f(x) = 1/2 * ||Wx - w||² + μ/2 * ||x||² 
    subject to    A * x = b

"""
# Define the problem data
Random.seed!(2)
n = 150
m = 100
d = 20 # number constraints
μ = 1e-1
# linear pieces
W = sprand(Float64, m, n, 0.3)
w = rand(Float64, m) * 2 .- 1
# constraints
A = sprand(Float64, d, n, 0.2)
b = A * rand(Float64, n)

f(x) = 0.5 * norm(W * x - w)^2 + 0.5 * μ * norm(x)^2
g(x) = W' * (W * x - w) + μ * x
H(x) = Symmetric(W' * W + μ * I)
hvp(x, v) = W' * (W * v) + μ * v
r(x) = A * x - b
Prob = LinearConstrProblem(f, H, g, A, b)


# Define JuMP model for original problem (to be solved by Gurobi)
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_attribute(model, "OutputFlag", 0)
@variable(model, var[1:n])
@constraint(model, A * var .== b)
@objective(model, Min, 0.5 * sum((W * var - w).^2) + 0.5 * μ * sum(var.^2))
optimize!(model)
println("Gurobi: optimal value = ", objective_value(model))


x0 = ones(n) / 10
y0 = ones(d) / 10

x, iter, x_history, L_history = HomALM(Prob, x0, y0, ρ; tol=1e-4, max_iter=40000)

# Plot the residuals of the iterates and the objective value
plt_x_history = reduce(hcat, x_history)
plt_res = [norm(r(plt_x_history[:, i])) for i in 1:iter+1]
plt_obj = [f(plt_x_history[:, i]) for i in 1:iter+1]

scatter(plt_res[end-5000:end], label = "Residuals", xlabel = "Iteration", ylabel = "Residual", title = "Residuals of HomALM")
scatter(plt_obj[end-5000:end], label = "Objective value", xlabel = "Iteration", ylabel = "Objective value", title = "Objective value of HomALM")