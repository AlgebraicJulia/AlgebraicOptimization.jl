using AlgebraicOptimization
using Distributions
using Random
using Graphs
using LinearAlgebra
include("paper-utils.jl")

Random.seed!(1234)

# Convergence vs. delay experiment
function generate_convergence_trajectories(file_path::String)
    n_agents = 20
    rm_generator(stalk_dim) = rand(1, stalk_dim)
    g = random_regular_graph(n_agents, 4)
    s = sheaf_from_graph(g, 4, rm_generator)
    L = sheaf_laplacian_matrix(s)
    K = opnorm(L, 2)
    γ = 1 / K
    x0 = rand(5.0:0.001:15.0, 4 * n_agents)
    T = Int(1e5)
    f = energy_function(s)
    x_orthog = nearest_global_section(s, x0)

    #Bs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Bs = [1, 50, 100, 500, 1000]
    for (i, B) in enumerate(Bs)
        update_d1 = Normal(0.05 * B, 0.005 * B)
        update_d2 = Normal(0.5 * B, 0.05 * B)
        update_weights = [0.8, 0.2]
        update_mixture = MixtureModelParams([update_d1, update_d2], update_weights)

        broadcast_d1 = Normal(0.1 * B, 0.01 * B)
        broadcast_d2 = Normal(0.8 * B, 0.05 * B)
        broadcast_weights = [0.5, 0.5]
        broadcast_mixture = MixtureModelParams([broadcast_d1, broadcast_d2], broadcast_weights)

        traj = compute_trajectory_asynch(L, x0, γ, n_agents, 4, update_mixture, broadcast_mixture; max_iters=T, B=B)
        loss = f.(traj)

        save_trajectory(file_path * "loss$i.csv", loss)

        # Postprocess for error
        x_star = traj[end]
        error(x) = norm(x - x_star)
        errors = error.(traj)
        # Truncate beyond error=0
        idx = findfirst(iszero, errors)
        if !isnothing(idx)
            errors = errors[1:idx-1]
        end

        save_trajectory(file_path * "error$i.csv", errors)

        # Postprocess for orthog_projection
        projection_error(x) = norm(x - x_orthog)
        projection_errors = projection_error.(traj)
        # Truncate beyond error=0
        idx = findfirst(iszero, projection_errors)
        if !isnothing(idx)
            projection_errors = projection_errors[1:idx-1]
        end

        save_trajectory(file_path * "proj_error$i.csv", projection_errors)
    end
end

function plot_convergence_losses(file_path::String)
    plt = empty_experiment_plot("")

    Bs = [1, 50, 100, 500, 1000]
    for (i, B) in enumerate(Bs)
        loss = load_trajectory(file_path * "loss$i.csv")
        if i == 1
            plot_log_loss_curve!(plt, loss, "B=0", colors[i])
        else
            plot_log_loss_curve!(plt, loss, "B=$B", colors[i])
        end

    end
    return plt
end

function plot_convergence_errors(file_path::String)
    plt = empty_experiment_plot(""; y_label="Error")
    Bs = Bs = [1, 50, 100, 500, 1000]
    for (i, B) in enumerate(Bs)
        loss = load_trajectory(file_path * "error$i.csv")
        if i == 1
            plot_log_loss_curve!(plt, loss, "B=0", colors[i])
        else
            plot_log_loss_curve!(plt, loss, "B=$B", colors[i])
        end

    end
    return plt
end

function plot_convergence_projection_errors(file_path::String)
    plt = empty_experiment_plot(""; y_label="Error")
    Bs = [1, 50, 100, 500, 1000]
    for (i, B) in enumerate(Bs)
        loss = load_trajectory(file_path * "proj_error$i.csv")
        T = 10000
        if length(loss) < T
            loss = vcat(loss, repeat([loss[end]], T - length(loss)))
        else
            loss = loss[1:T]
        end
        if i == 1
            plot_loss_curve!(plt, loss, "B=0", colors[i])
        else
            plot_loss_curve!(plt, loss, "B=$B", colors[i%5+1])
        end

    end
    return plt
end

function convergence_vs_delay_plot(file_path::String)


end

# Convergence from many initializations experiment


