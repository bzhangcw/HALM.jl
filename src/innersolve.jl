using KrylovKit

abstract type InnerSolve end

struct NewtonCG <: InnerSolve end

function search_direction(opt_inner::NewtonCG, x, gradx, hvp; rtol=1e-5)
    d, _ = KrylovKit.linsolve(
        (v -> hvp(x, v)), -gradx, zeros(n), rtol=rtol, issymmetric=true
    )
    return d
end