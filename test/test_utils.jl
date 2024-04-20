include("HomALM.jl")


function test_fun(Prob, x0, y0, ρ, model, subprob_model, fex; run_ALM::Bool = true)
    ## Method 0: Gurobi 
    optimize!(model)
    println("Gurobi: x = ", value.(var))

    ## Method 1: ALM
    if run_ALM
        x, iter = ALM(Prob, x0, y0, ρ, subprob_model, fex)
        println("ALM: x = ", x, ", iter = ", iter)
    end

    ## Method 2: Homogenized ALM
    x, iter, x_history, L_history = HomALM(Prob, x0, y0, ρ; tol=1e-4, max_iter=1500)
    println("HomALM: x = ", x, ", iter = ", iter)

    ## Plot the iterates for HomALM
    plt_x_history = reduce(hcat, x_history)
    # Compute the limits of the plot
    xmin = minimum(plt_x_history[1, :]) < 0 ? 1.1 * minimum(plt_x_history[1, :]) : 0.9 * minimum(plt_x_history[1, :])
    ymin = minimum(plt_x_history[2, :]) < 0 ? 1.1 * minimum(plt_x_history[2, :]) : 0.9 * minimum(plt_x_history[2, :])
    xmax = maximum(plt_x_history[1, :]) < 0 ? 0.9 * maximum(plt_x_history[1, :]) : 1.1 * maximum(plt_x_history[1, :])
    ymax = maximum(plt_x_history[2, :]) < 0 ? 0.9 * maximum(plt_x_history[2, :]) : 1.1 * maximum(plt_x_history[2, :])
    
    gif = @gif for i ∈ 1:iter
        scatter(plt_x_history[1, [i]], plt_x_history[2, [i]], markersize = 5, label = "Iterates")
        plot!(plt_x_history[1, 1:i], plt_x_history[2, 1:i], linestyle = :dash, label = "Iterates path")
        xlabel!("x₁")
        ylabel!("x₂")
        xlims!(xmin, xmax)
        ylims!(ymin, ymax)
        title!("Iterates of HomALM (iter = $i)")
    end every 5

    return gif
end