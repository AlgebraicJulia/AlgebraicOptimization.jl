using AlgebraicOptimization
using Distributions
using Random
using Graphs
using LinearAlgebra
include("paper-utils.jl")

#Random.seed!(420)
#Random.seed!(6)
Random.seed!(8)

const data_path = "./examples/ACC26-examples/data/"

# Singular value experiment
function run_sv_experiment()


end

# Orthogonal projection experiment
function run_op_experiment(file_path)
    n_agents = 20
    g = random_regular_graph(n_agents, 4)
    s = sheaf_from_graph(g, 4, dim -> random_semi_orthogonal_matrix(1, dim))
    L = sheaf_laplacian_matrix(s)
    K = opnorm(L, 2)
    γ = 1 / K
    T = 1e5

    x0 = rand(Normal(0, 10), 4 * n_agents)
    x_star = nearest_global_section(s, x0)

    results = Float64[]

    for i in 1:15
        B = 2^(i - 1)
        # 3 trials for each B
        sols = []
        for _ in 1:3
            traj = compute_trajectory_asynch(L, x0, γ, n_agents, 4, ones(Int, n_agents), repeat([B], n_agents); B=B, max_iters=T)
            push!(sols, traj[end])
        end
        # Take the average distance from x_star
        r = sum([norm(s - x_star) for s in sols]) / 3
        push!(results, r)
    end
    save_trajectory(file_path * "results.csv", results)
end

function plot_op_experiment(file_path)
    results = load_trajectory(file_path * "results.csv")
    #results = results[2:end]
    Bs = [2^(i - 1) for i in 1:15]
    #Bs = Bs[2:end]
    xs = collect(1:14)
    #Bs[1] = 0

    plt = scatter(Bs, results, xscale=:log2, color=rgb(179, 163, 105), xlabel=L"B+1", ylabel=L"||\mathbf{x}^* - \mathbf{x}(0)^\bot||", thickness_scaling=1.9)

    return plt
end


# Many initializations experiment
function run_many_initializations(N)
    n_agents = 20
    g = random_regular_graph(n_agents, 4)
    s = sheaf_from_graph(g, 4, dim -> random_semi_orthogonal_matrix(1, dim))
    L = sheaf_laplacian_matrix(s)
    K = opnorm(L, 2)
    γ = 1 / K
    B = 50
    #x0 = rand(5.0:0.001:15.0, 4 * n_agents)
    T = 1e5
    f = energy_function(s)

    update_d1 = Normal(0.05 * B, 0.005 * B)
    update_d2 = Normal(0.5 * B, 0.05 * B)
    update_weights = [0.5, 0.5]
    update_mixture = MixtureModelParams([update_d1, update_d2], update_weights)

    broadcast_d1 = Normal(0.1 * B, 0.01 * B)
    broadcast_d2 = Normal(0.8 * B, 0.05 * B)
    broadcast_weights = [0.5, 0.5]
    broadcast_mixture = MixtureModelParams([broadcast_d1, broadcast_d2], broadcast_weights)

    x0s = [rand(-10:0.01:10, 4 * n_agents) for _ in 1:N]
    trajs = compute_trajectory_asynch(L, x0s, γ, n_agents, 4, update_mixture, broadcast_mixture; B=B, max_iters=T)

    energy_plt = empty_experiment_plot(; xlabel=L"\textnormal{Iteration, } t", ylabel=L"f(\mathbf{x})")
    error_plt = empty_experiment_plot(; xlabel=L"\textnormal{Iteration, } t", ylabel=L"\frac{||\mathbf{x}-\mathbf{x}^*||}{||\mathbf{x}^*||}")
    norm_plt = empty_experiment_plot(; xlabel=L"\textnormal{Iteration, } t", ylabel=L"||\mathbf{x}||")
    for traj in trajs
        energies = f.(traj)
        x_star = nearest_global_section(s, traj[end])
        rel_error(x) = norm(x - x_star) / norm(x_star)

        errors = rel_error.(traj)
        norms = norm.(traj)
        plot_log_loss_curve!(energy_plt, energies, ""; color=blue, alpha=0.2)
        plot_log_loss_curve!(error_plt, errors, ""; color=orange, alpha=0.2)
        plot_loss_curve!(norm_plt, norms, ""; color=green, alpha=0.2)
    end
    l = @layout [a b c]
    return plot(energy_plt, error_plt, norm_plt, layout=l, size=(1800, 500), thickness_scaling=2)
end

# Stepsize vs. Convergence experiment
function generate_stepsize_trajectories(file_path::String)
    # Run experiment with synchronous stepsize
    n_agents = 20
    #rm_generator(stalk_dim) = rand(1, stalk_dim)
    #rm_generator(stalk_dim) = Matrix{Float64}(I(stalk_dim))
    g = random_regular_graph(n_agents, 4)
    s = sheaf_from_graph(g, 4, dim -> random_semi_orthogonal_matrix(2, dim))
    #s = sheaf_from_graph(g, 4, matrix_weighted_edge_generator(pd_prob=0.8); symmetric_edges=true)
    L = sheaf_laplacian_matrix(s)
    K = opnorm(L, 2)
    γ1 = 1 / K
    B = 50
    γ2 = 0.1 * γ1
    #γ2 = 2 / (K * (1 + 2 * (sqrt(n_agents) * B)))
    x0 = rand(5.0:0.001:15.0, 4 * n_agents)
    T = 1e6
    f = energy_function(s)

    update_d1 = Normal(0.05 * B, 0.005 * B)
    update_d2 = Normal(0.5 * B, 0.05 * B)
    update_weights = [0.8, 0.2]
    update_mixture = MixtureModelParams([update_d1, update_d2], update_weights)

    broadcast_d1 = Normal(0.1 * B, 0.01 * B)
    broadcast_d2 = Normal(0.8 * B, 0.05 * B)
    broadcast_weights = [0.5, 0.5]
    broadcast_mixture = MixtureModelParams([broadcast_d1, broadcast_d2], broadcast_weights)

    trajs = compute_trajectory_asynch(L, x0, [γ1, γ2], n_agents, 4, update_mixture, broadcast_mixture; max_iters=T, B=B)
    losses = [f.(traj) for traj in trajs]

    save_trajectory(file_path * "loss_synch_ss.csv", losses[1])
    save_trajectory(file_path * "loss_asynch_ss.csv", losses[2])

    traj = compute_trajectory(L, x0, 1 / K; max_iters=T)
    loss = f.(traj)
    save_trajectory(file_path * "loss_synch.csv", loss)

end

function plot_stepsize_losses(file_path::String)
    plt = empty_experiment_plot("Iteration", "Energy")
    loss_synch = load_trajectory(file_path * "loss_synch.csv")
    loss_synch_ss = load_trajectory(file_path * "loss_synch_ss.csv")
    loss_asynch_ss = load_trajectory(file_path * "loss_asynch_ss.csv")

    plot_log_loss_curve!(plt, loss_synch, "Synch 1/K")
    plot_log_loss_curve!(plt, loss_synch_ss, "Asynch 1/K")
    plot_log_loss_curve!(plt, loss_asynch_ss, "Asynch 1/(K*B)")
    return plt
end


# Convergence vs. delay experiment
function generate_convergence_trajectories(file_path::String, s::EuclideanSheaf)
    #n_agents = 20
    n_agents = length(vertex_stalks(s))
    #rm_generator(stalk_dim) = rand(1, stalk_dim)
    #rm_generator(stalk_dim) = Matrix{Float64}(I(stalk_dim))
    #g = random_regular_graph(n_agents, 4)
    #s = sheaf_from_graph(g, 4, rm_generator)
    #s = sheaf_from_graph(g, 4, dim -> random_semi_orthogonal_matrix(2, dim))
    #s = sheaf_from_graph(g, 4, matrix_weighted_edge_generator(pd_prob=0.8); symmetric_edges=true)
    L = sheaf_laplacian_matrix(s)
    K = opnorm(L, 2)
    γ = 1 / K
    x0 = rand(5.0:0.001:15.0, 4 * n_agents)
    T = Int(1e5)
    f = energy_function(s)
    x_orthog = nearest_global_section(s, x0)

    #Bs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #Bs = [1, 50, 100, 250, 500]
    Bs = [1, 10, 50, 100, 200]
    for (i, B) in enumerate(Bs)
        update_d1 = Normal(0.05 * B, 0.005 * B)
        update_d2 = Normal(0.5 * B, 0.05 * B)
        update_weights = [0.5, 0.5]
        update_mixture = MixtureModelParams([update_d1, update_d2], update_weights)

        broadcast_d1 = Normal(0.1 * B, 0.01 * B)
        broadcast_d2 = Normal(0.8 * B, 0.05 * B)
        broadcast_weights = [0.5, 0.5]
        broadcast_mixture = MixtureModelParams([broadcast_d1, broadcast_d2], broadcast_weights)

        traj = compute_trajectory_asynch(L, x0, γ, n_agents, 4, update_mixture, broadcast_mixture; max_iters=T, B=B)
        loss = f.(traj)

        save_trajectory(file_path * "loss$i.csv", loss)

        # Postprocess for error
        x_star = nearest_global_section(s, traj[end])
        println(norm(x_star))
        error(x) = norm(x - x_star) / norm(x_star)
        errors = error.(traj)
        #errors .+= 1e-12
        # Truncate beyond error=1e-6
        #=idx = findfirst(e -> e < 1e-6, errors)
        if !isnothing(idx)
            errors = errors[1:idx-1]
        end=#

        save_trajectory(file_path * "error$i.csv", errors)

        # Postprocess for orthog_projection
        projection_error(x) = norm(x - x_orthog) / norm(x_orthog)
        projection_errors = projection_error.(traj)
        # Truncate beyond error=0
        #=idx = findfirst(iszero, projection_errors)
        if !isnothing(idx)
            projection_errors = projection_errors[1:idx-1]
        end=#

        save_trajectory(file_path * "proj_error$i.csv", projection_errors)
    end
end

function run_rm_experiment(file_path::String)
    n_agents = 20
    stalk_dim = 4
    g = random_regular_graph(n_agents, 4)
    #g = Graphs.grid([5, 5])
    #g = binary_tree(5)
    #g = cycle_graph(n_agents)
    #g = path_graph(n_agents)
    rm1_generator(stalk_dim) = random_semi_orthogonal_matrix(stalk_dim, stalk_dim)
    rm2_generator(stalk_dim) = Matrix{Float64}(I(stalk_dim))

    rm3_generator = matrix_weighted_edge_generator(; pd_prob=0.8)
    rm4_generator(stalk_dim) = rand(1, stalk_dim)

    sheaves = [
        sheaf_from_graph(g, stalk_dim, rm2_generator), # Constant sheaf
        sheaf_from_graph(g, stalk_dim, rm4_generator), # Random sheaf
        sheaf_from_graph(g, stalk_dim, rm3_generator; symmetric_edges=true) # Matrix weighted consensus sheaf
    ]
    paths = [file_path * "constant_sheaf/", file_path * "vector_bundle/", file_path * "matrix_weighted/"]
    for (s, p) in zip(sheaves, paths)
        generate_convergence_trajectories(p, s)
    end
end

function rm_experiment_plot(file_path::String)
    p1 = plot_convergence_losses(file_path * "constant_sheaf/"; legend=true, ylabel=L"f(\mathbf{x}(t))")
    p2 = plot_convergence_errors(file_path * "constant_sheaf/"; ylabel=L"\frac{||\mathbf{x}(t)-\mathbf{x}^*||}{||\mathbf{x}(t)^*||}")
    p3 = plot_convergence_projection_errors(file_path * "constant_sheaf/"; xlabel=L"\textnormal{Iteration, } t", ylabel=L"\frac{||\mathbf{x}(t)-\Pi_\Gamma[\mathbf{x}(0)]||}{||\Pi_\Gamma[\mathbf{x}(0)]||}")
    p4 = plot_convergence_losses(file_path * "vector_bundle/")#; xlabel="", ylabel="")
    p5 = plot_convergence_errors(file_path * "vector_bundle/")
    p6 = plot_convergence_projection_errors(file_path * "vector_bundle/"; xlabel=L"\textnormal{Iteration, } t")#, ylabel="")
    p7 = plot_convergence_losses(file_path * "matrix_weighted/")#; xlabel="", ylabel="")
    p8 = plot_convergence_errors(file_path * "matrix_weighted/")
    p9 = plot_convergence_projection_errors(file_path * "matrix_weighted/"; xlabel=L"\textnormal{Iteration, } t")#, ylabel="")
    l = @layout [a b c; d e f; g h i]
    plt = plot(p1, p4, p7, p2, p5, p8, p3, p6, p9, layout=l, size=(1900, 1000), title=["I" "II" "III" "" "" "" "" "" ""])
    savefig(plt, "rm_experiment.png")
end

function generate_all_convergence_trajectories(file_path::String)
    # Generate 3 test cases
    n_agents = 20
    gs = [random_regular_graph(20, 4), erdos_renyi(20, 0.3), star_graph(20)]
    rm_generator(stalk_dim) = rand(1, stalk_dim)
    sheaves = [sheaf_from_graph(g, 4, rm_generator) for g in gs]
    paths = [file_path * "regular_graph/", file_path * "er_graph/", file_path * "complete_graph/"]
    for (s, p) in zip(sheaves, paths)
        generate_convergence_trajectories(p, s)
    end
end

function plot_convergence_losses(file_path::String; kwargs...)
    plt = empty_experiment_plot(; kwargs...)

    #Bs = [1, 50, 100, 250, 500]
    Bs = [1, 10, 50, 100, 200]
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

function plot_convergence_errors(file_path::String; kwargs...)
    plt = empty_experiment_plot(; kwargs...)

    #Bs = [1, 50, 100, 250, 500]
    Bs = [1, 10, 50, 100, 200]
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

function plot_convergence_projection_errors(file_path::String; kwargs...)
    plt = empty_experiment_plot(; kwargs...)
    Bs = [1, 10, 50, 100, 200]
    for (i, B) in enumerate(Bs)
        loss = load_trajectory(file_path * "proj_error$i.csv")
        #=T = 10000
        if length(loss) < T
            loss = vcat(loss, repeat([loss[end]], T - length(loss)))
        else
            loss = loss[1:T]
        end=#
        if i == 1
            plot_log_loss_curve!(plt, loss, "B=0")
        else
            plot_log_loss_curve!(plt, loss, "B=$B")
        end

    end
    return plt
end

function convergence_vs_delay_plot(file_path::String)
    p1 = plot_convergence_losses(file_path * "regular_graph/"; legend=true, ylabel=L"f(\mathbf{x})")
    p2 = plot_convergence_errors(file_path * "regular_graph/"; xlabel=L"\mathrm{Iteration},\quad t", ylabel=L"\frac{||\mathbf{x}-\mathbf{x}^*||}{||\mathbf{x}^*||}")
    p3 = plot_convergence_losses(file_path * "er_graph/")#; xlabel="", ylabel="")
    p4 = plot_convergence_errors(file_path * "er_graph/"; xlabel=L"\mathrm{Iteration},\quad t")#, ylabel="")
    p5 = plot_convergence_losses(file_path * "complete_graph/")#; xlabel="", ylabel="")
    p6 = plot_convergence_errors(file_path * "complete_graph/"; xlabel=L"\mathrm{Iteration},\quad t")#, ylabel="")
    l = @layout [a b c; d e f]
    plot(p1, p3, p5, p2, p4, p6, layout=l, size=(1900, 700))
end

#generate_stepsize_trajectories(data_path * "ss_vs_converge/")
#plot_stepsize_losses(data_path * "ss_vs_converge/")

#generate_all_convergence_trajectories(data_path * "converge_vs_delay/")
#plot_convergence_losses(data_path * "converge_vs_delay/")
#convergence_vs_delay_plot(data_path * "converge_vs_delay/")

#run_op_experiment(data_path * "op_experiment/")
#plot_op_experiment(data_path * "op_experiment/")


#run_rm_experiment(data_path * "rm_experiment/")
#rm_experiment_plot(data_path * "rm_experiment/")
#run_many_initializations(100)
function test_sigma2()
    sigma2s = Float64[]
    iters_to_converge = Int64[]
    for i in 1:100
        g = erdos_renyi(20, 0.3)
        if !is_connected(g)
            continue
        end
        #g = random_regular_graph(20, 4)
        s = sheaf_from_graph(g, 4, matrix_weighted_edge_generator(pd_prob=0.5); symmetric_edges=true)
        L = sheaf_laplacian_matrix(s)
        K = opnorm(L)
        γ = 1 / K
        vals, vecs = eigen(L)
        sigma2_idx = findfirst(x -> x > 1e-12, vals)
        sigma2 = vals[sigma2_idx]
        if sigma2 < 0.06
            continue
        end
        #println(vals[sigma2_idx])
        B = 50
        T = 1e5

        x0 = rand(Normal(0, 10), 80)
        traj = compute_trajectory_asynch(L, x0, γ, 20, 4, repeat([10], 20), repeat([B], 20); max_iters=T)
        println("σ2 = $sigma2: Converged in $(length(traj)) iterations.")
        if length(traj) > T
            continue
        end
        push!(sigma2s, sigma2)
        push!(iters_to_converge, length(traj))
    end
    save_trajectory(data_path * "sigma2/sigma2s.csv", sigma2s)
    save_trajectory(data_path * "sigma2/iters_to_converge.csv", iters_to_converge)
    scatter(sigma2s, iters_to_converge)
end

function plot_sigma2()
    sigma2s = load_trajectory(data_path * "sigma2/sigma2s.csv")
    iters_to_converge = load_trajectory(data_path * "sigma2/iters_to_converge.csv")

    plt = empty_experiment_plot(ylabel=L"t^*", xlabel=L"\lambda_2")
    scatter!(plt, sigma2s, iters_to_converge, color=blue)
    return plt
end


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