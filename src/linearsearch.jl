abstract type LineSearch end

struct backtrack <: LineSearch end

"""

linearsearch(opt_ls::backtrack, L::Function, x, d, prev_L; Kₗ = 20, α₀ = 10.0)

Perform a line search using the backtrack algorithm.

Inputs:
- `opt_ls::backtrack` is the line search option.
- `L::Function` is the objective function.
- `x` is the current iterate.
- `d` is the search direction (need to be a descent direction to guarantee the existence of α).
- `prev_L` is the objective function value at the previous iterate.

Optional inputs:
- `Kₗ` is the maximum number of iterations.
- `α₀` is the initial step size.
"""
function linearsearch(opt_ls::backtrack, L::Function, x, d, prev_L; Kₗ = 20, α₀ = 10.0)
    α = α₀
    dₙ = d |> norm
    kₗ = 0
    while kₗ < Kₗ && α > 1e-4
        Lx = L(x + d * α)
        if Lx - prev_L <= -1e-1 * α * dₙ
            break
        end
        α *= 0.5
        kₗ += 1
    end
    return α, kₗ
end