using AlgebraicOptimization
using Distributions
using Random
using Graphs
using LinearAlgebra
include("paper-utils.jl")

Random.seed!(420)

const data_path = "./examples/ACC26-examples/data/converge_vs_delay/"



# Convergence vs. delay experiment
function generate_convergence_trajectories(file_path::String)
    n_agents = 20
    #rm_generator(stalk_dim) = rand(1, stalk_dim)
    #rm_generator(stalk_dim) = Matrix{Float64}(I(stalk_dim))
    g = random_regular_graph(n_agents, 4)
    s = sheaf_from_graph(g, 4, dim -> random_semi_orthogonal_matrix(2, dim))
    #s = sheaf_from_graph(g, 4, matrix_weighted_edge_generator(pd_prob=0.8); symmetric_edges=true)
    L = sheaf_laplacian_matrix(s)
    K = opnorm(L, 2)
    γ = 1 / K
    x0 = rand(5.0:0.001:15.0, 4 * n_agents)
    T = Int(1e6)
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
        error(x) = norm(x - x_star) / norm(x_star)
        errors = error.(traj[1:end-1])
        #errors .+= 1e-12
        # Truncate beyond error=1e-6
        idx = findfirst(e -> e < 1e-6, errors)
        if !isnothing(idx)
            errors = errors[1:idx-1]
        end

        save_trajectory(file_path * "error$i.csv", errors)

        # Postprocess for orthog_projection
        #=projection_error(x) = norm(x - x_orthog)
        projection_errors = projection_error.(traj)
        # Truncate beyond error=0
        #=idx = findfirst(iszero, projection_errors)
        if !isnothing(idx)
            projection_errors = projection_errors[1:idx-1]
        end=#

        save_trajectory(file_path * "proj_error$i.csv", projection_errors)=#
    end
end

function plot_convergence_losses(file_path::String)
    plt = empty_experiment_plot("Iteration", "Energy")

    Bs = [1, 50, 100, 500, 1000]
    for (i, B) in enumerate(Bs)
        loss = load_trajectory(file_path * "loss$i.csv")
        if i == 1
            plot_log_loss_curve!(plt, loss, "B=0")
        else
            plot_log_loss_curve!(plt, loss, "B=$B")
        end

    end
    return plt
end

function plot_convergence_errors(file_path::String)
    plt = empty_experiment_plot("Iteration", "Error"; legend=false)
    Bs = Bs = [1, 50, 100, 500, 1000]
    for (i, B) in enumerate(Bs)
        loss = load_trajectory(file_path * "error$i.csv")
        if i == 1
            plot_log_loss_curve!(plt, loss, "B=0")
        else
            plot_log_loss_curve!(plt, loss, "B=$B")
        end

    end
    return plt
end

function plot_convergence_projection_errors(file_path::String)
    plt = empty_experiment_plot("", "Error")
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
    p1 = plot_convergence_losses(file_path)
    p2 = plot_convergence_errors(file_path)
    l = @layout [a; b]
    plot(p1, p2, layout=l)
end



generate_convergence_trajectories(data_path)
plot_convergence_losses(data_path)
#convergence_vs_delay_plot(data_path)
#=
n_agents = 20
#rm_generator(stalk_dim) = rand(3, stalk_dim)

#rm_generator(stalk_dim) = Matrix{Float64}(I(stalk_dim))
g = random_regular_graph(n_agents, 4)
#s = sheaf_from_graph(g, 4, matrix_weighted_edge_generator(pd_prob=0.0); symmetric_edges=true)
s = sheaf_from_graph(g, 4, dim -> random_semi_orthogonal_matrix(2, dim))
L = sheaf_laplacian_matrix(s)
K = opnorm(L, 2)
γ = 1 / K

vals, vecs = eigen(collect(I(80) - γ * L))

plt = histogram(vals; bins=80)

x0 = rand(4 * n_agents)

traj_synch = compute_trajectory(L, x0, γ; max_iters=5e6)
losses_synch = energy_function(L).(traj_synch)
B = 10
update_d1 = Normal(0.05 * B, 0.005 * B)
update_d2 = Normal(0.5 * B, 0.05 * B)
update_weights = [0.8, 0.2]
update_mixture = MixtureModelParams([update_d1, update_d2], update_weights)

broadcast_d1 = Normal(0.1 * B, 0.01 * B)
broadcast_d2 = Normal(0.8 * B, 0.05 * B)
broadcast_weights = [0.5, 0.5]
broadcast_mixture = MixtureModelParams([broadcast_d1, broadcast_d2], broadcast_weights)
traj_asynch = compute_trajectory_asynch(L, x0, γ, n_agents, 4, update_mixture, broadcast_mixture; B=B, max_iters=5e6)
losses_asynch = energy_function(L).(traj_asynch)
plt = plot(losses_synch[1:10:end]; yscale=:log10, label="Synch")
plot!(plt, losses_asynch[1:10:end]; yscale=:log10, label="Asynch")=#

# Convergence from many initializations experiment