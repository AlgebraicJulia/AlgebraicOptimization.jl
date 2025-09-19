using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays
using Graphs
using Distributions
using YAML

# Structure to hold seeds for all random components of the experiment
struct ExperimentSeeds
    graph_seed::Int
    sheaf_seed::Int
    init_cond_seed::Int
    delay_seed::Int
end

"""
Generate a new set of random seeds.
"""
function generate_seeds()
    return ExperimentSeeds(
        rand(1:1_000_000),
        rand(1:1_000_000),
        rand(1:1_000_000),
        rand(1:1_000_000)
    )
end


"""
Generates a random orthogonal matrix.
"""
function random_orthogonal_matrix(rng::AbstractRNG, dim::Int)
    A = randn(rng, dim, dim)
    return Matrix(qr(A).Q)
end

"""
Generates a random positive semi-definite matrix.
"""
function random_psd_matrix(rng::AbstractRNG, dim::Int)
    B = randn(rng, dim, dim)
    return B' * B
end

function block_normalize_matrix(L::AbstractMatrix, num_blocks::Int, block_size::Int; tol=1e-9::Float64)
    total_dim = num_blocks * block_size
    @assert size(L) == (total_dim, total_dim) "Matrix dimensions must match the block structure."
    D = zeros(total_dim, total_dim)
    for i in 1:num_blocks
        idx_range = (i - 1) * block_size + 1 : i * block_size
        D[idx_range, idx_range] = L[idx_range, idx_range]
    end
    D_inv_sqrt = zeros(total_dim, total_dim)
    for i in 1:num_blocks
        idx_range = (i - 1) * block_size + 1 : i * block_size
        block = D[idx_range, idx_range]
        if norm(block) > tol
            symmetric_block = (block + block') / 2
            D_inv_sqrt[idx_range, idx_range] = pinv(sqrt(symmetric_block))
        end
    end
    return D_inv_sqrt * L * D_inv_sqrt
end

function smallest_nonzero_eigenvalue(M::AbstractMatrix; tol=1e-9)
    if !issymmetric(M)
        @warn "Matrix is not symmetric. Eigenvalues may be complex."
    end
    
    eigenvalues = eigvals(Symmetric(M))
    nonzero_eigenvalues = filter(e -> abs(e) > tol, eigenvalues)
    if isempty(nonzero_eigenvalues)
        return nothing
    else
        return minimum(abs.(nonzero_eigenvalues))
    end
end

"""
Sheaf generator that creates a sheaf from a graph with specified properties.
Randomness is controlled by the provided RNG.
"""
function sheaf_from_graph(rng::AbstractRNG, g::Graph, vertex_dim::Int, edge_dim::Int; style::String=nothing)
    s = EuclideanSheaf{Float64}(repeat([vertex_dim], nv(g)))

    rm_bundle(rng_inner::AbstractRNG, vd::Int, ed::Int) = Matrix{Float64}(randn(rng_inner) * random_orthogonal_matrix(rng_inner, vd))
    rm_random(rng_inner::AbstractRNG, vd::Int, ed::Int) = Matrix{Float64}(randn(rng_inner, ed, vd))
    rm_identity(rng_inner::AbstractRNG, vd::Int, ed::Int) = Matrix{Float64}(I(vd))
    rm_weighted(rng_inner::AbstractRNG, vd::Int, ed::Int) = Matrix{Float64}(random_psd_matrix(rng_inner, vd))

    for e in edges(g)
        i, j = src(e), dst(e)
        rm1, rm2 = if style == "weighted"
            if vertex_dim != edge_dim
                error("Type Error: vertex dimension must equal edge_dim for a matrix-weighted graph")
            end
            W = rm_weighted(rng, vertex_dim, edge_dim)
            (W, W)
        elseif style == "bundle"
            if vertex_dim != edge_dim
                error("Type Error: vertex dimension must equal edge_dim for a discrete vector bundle")
            end
            (rm_bundle(rng, vertex_dim, edge_dim), rm_identity(rng, vertex_dim, edge_dim))
        elseif style == "constant"
            if vertex_dim != edge_dim
                error("Type Error: vertex dimension must equal edge_dim for a constant sheaf")
            end
            (rm_identity(rng, vertex_dim, edge_dim), rm_identity(rng, vertex_dim, edge_dim))
        elseif style == "random"
            (rm_random(rng, vertex_dim, edge_dim), rm_random(rng, vertex_dim, edge_dim))
        end
        add_sheaf_edge!(s, i, j, rm1, rm2)
    end
    return s
end

energy_function(L) = x -> 0.5 * x' * (L * x) # Dirichlet energy function

"""
Computes the trajectory of the system under asynchronous updates.
Randomness of delays is controlled by the provided RNG.
Returns the loss history and the state history.
"""
function compute_trajectory(rng::AbstractRNG, L, x0, γ, num_blocks, block_size; B_min=1, B_max=1, max_iters=1000, tol=1e-8)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], num_blocks))
    local_states = BlockArray(hcat([global_state for _ in 1:num_blocks]...), repeat([block_size], num_blocks), ones(Int, num_blocks))
    
    state_history = [Vector(global_state)]
    losses = [f(global_state)]

    periods = rand(rng, B_min:B_max, num_blocks)
    phases = [rand(rng, 0:periods[i]-1) for i in 1:num_blocks]

    for t in 1:max_iters
        g = BlockArray(L * local_states, repeat([block_size], num_blocks), ones(Int, num_blocks)) # every agent computes a local update
        for i in 1:num_blocks
            if t % periods[i] == phases[i]
                local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)] # update local state
                global_state[Block(i)] = local_states[Block(i), Block(i)][:] # update global state
                x = local_states[Block(i), Block(i)]  # if it's the right time, broadcast your local state to other agents

                local_states[Block(i), :] .= x
            end
        end
        push!(state_history, Vector(global_state))
        push!(losses, f(global_state))
        if losses[end] < tol || losses[end] > 1e10 # Stop if it converges or diverges wildly
            break
        end
    end
    return losses, state_history
end

"""
Calculates the beta(t) metric from the paper.
"""
function calculate_beta(state_history, B)
    beta_trajectory = []
    for t in 1:length(state_history)
        start_idx = max(1, t - B - 1)
        end_idx = max(1, t - 1)
        
        sum_sq_diff = 0.0
        for τ in start_idx:end_idx
            if τ + 1 <= length(state_history)
                sum_sq_diff += norm(state_history[τ+1] - state_history[τ])^2
            end
        end
        push!(beta_trajectory, sum_sq_diff)
    end
    return beta_trajectory
end


"""
Main experiment runner function.
Takes a seeds struct to ensure reproducibility.
"""
function run_experiment(config,seeds::ExperimentSeeds)
    """
    Sheaf parameters
    """
    N =  config["graph"]["num_nodes"] # number of agents
    degree = config["graph"]["degree"] # degree of the graph
    vertex_dim = config["sheaf"]["vertex_dim"] # vertex stalk dimension
    edge_dim = config["sheaf"]["edge_dim"] # edge stalk dimension
    sheaf_type = config["sheaf"]["type"] # choices: "bundle", "weighted", "constant"

    """
    Generate sheaf and normalized sheaf Laplacian
    """
    graph_rng = MersenneTwister(seeds.graph_seed)
    g = random_regular_graph(N, degree; rng=graph_rng) # random graph

    sheaf_rng = MersenneTwister(seeds.sheaf_seed)
    s = sheaf_from_graph(sheaf_rng, g, vertex_dim, edge_dim; style=sheaf_type) # sheaf
    
    L_unnormalized = sheaf_laplacian_matrix(s)
    if config["sheaf"]["normalize"]
        L =  block_normalize_matrix(L_unnormalized, N, vertex_dim) # sheaf Laplacian
    else
        L = L_unnormalized
    end
    K = opnorm(L, 2) # spectral radius (Lipschitz constant K)
    η = smallest_nonzero_eigenvalue(L) # Frustration (related to PL constant)

    """
    Experiment parameters
    """
    init_cond_rng = MersenneTwister(seeds.init_cond_seed)
    x0 = randn(init_cond_rng, vertex_dim * N) # initial condition

    B_min = 1
    B_max = 1
    if !isnothing(config["alg"]["delay_min"])
        B_min = config["alg"]["delay_min"]
    end
    if !isnothing(config["alg"]["delay_max"])
        B_max = config["alg"]["delay_max"]
    end
    
    if !isnothing(config["alg"]["step-size"])
        γ = config["alg"]["step-size"]
    else
        γ = 1 / K 
    end

    println("Running experiment with seeds: ", seeds)
    println("number of agents: ", N)
    println("step-size (γ): ", γ)
    println("spectral radius (K): ", K)
    println("frustration (η): ", η)
    println("Max delay (B): ", B_max)
    println("\n")

    delay_rng_sync = MersenneTwister(seeds.delay_seed)
    delay_rng_async = MersenneTwister(seeds.delay_seed)

    alpha_sync, sync_history = compute_trajectory(delay_rng_sync, L, x0, γ, N, vertex_dim; max_iters=config["alg"]["num_iters"])
    alpha_async, async_history = compute_trajectory(delay_rng_async, L, x0, γ, N, vertex_dim; B_min=B_min, B_max=B_max, max_iters=config["alg"]["num_iters"])

    beta_sync = calculate_beta(sync_history, 0) # B=0 for synchronous case
    beta_async = calculate_beta(async_history, B_max)

    plot_epsilon = 1e-16 # Small constant to avoid log(0)

    p1 = plot(yscale=:log10, title="Sync v.s. Async", xlabel="t", ylabel="alpha(t)", ylims=(plot_epsilon, Inf))
    plot!(p1, alpha_sync .+ plot_epsilon, label="sync", color=:blue, alpha=0.75)
    plot!(p1, alpha_async .+ plot_epsilon, label="async", color=:orange, alpha=0.75)
    
    p2 = plot(yscale=:log10, title="", xlabel="t", ylabel="beta(t)", ylims=(plot_epsilon, Inf))
    plot!(p2, beta_sync .+ plot_epsilon, label="", color=:blue, alpha=0.75)
    plot!(p2, beta_async .+ plot_epsilon, label="", color=:orange, alpha=0.75)

    display(plot(p1, p2, layout = (2,1)))

    return (alpha_sync, alpha_async), (beta_sync, beta_async)
end

"""
Searches for a set of seeds that causes asynchronous divergence.
"""
function find_divergent_example(config;max_attempts=1000)
    for attempt in 1:max_attempts
        seeds = generate_seeds()
        (alpha_sync, alpha_async), _ = run_experiment(config,seeds)

        if alpha_sync[end] < 1e-5 && alpha_async[end] > alpha_async[1]
            println("Found divergent example after $attempt attempts!")
            return seeds
        end
        if attempt % 100 == 0
            println("Attempt $attempt")
        end
    end
    println("Could not find a divergent example after $max_attempts attempts.")
    return nothing
end

config = YAML.load_file(joinpath(@__DIR__, "config.yaml"))
seeds = generate_seeds()
_ = run_experiment(config,seeds) # Assign to _ to suppress output
println("Experiment finished. Plots are displayed above.")

#=
# Uncomment to search for a new divergent example
divergent_seeds = find_divergent_example(config)

if !isnothing(divergent_seeds)
    run_experiment(config,divergent_seeds)
else
    println("No divergent example found. Try running find_divergent_example() again.")
end
=#

