using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays
using Graphs
using Distributions

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
        else
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
"""
function compute_trajectory(rng::AbstractRNG, L, x0, γ, num_blocks, block_size; B_min=1, B_max=1, max_iters=1000, tol=1e-8)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], num_blocks))
    local_states = BlockArray(hcat([global_state for _ in 1:num_blocks]...), repeat([block_size], num_blocks), ones(Int, num_blocks))

    periods = rand(rng, B_min:B_max, num_blocks)
    phases = [rand(rng, 0:periods[i]-1) for i in 1:num_blocks]
    losses = [f(x0)]

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
        push!(losses, f(global_state))
        if losses[end] < tol
            break
        end
    end
    return losses, global_state
end

"""
Main experiment runner function.
Takes a seeds struct to ensure reproducibility.
"""
function run_experiment(seeds::ExperimentSeeds)
    """
    Sheaf parameters
    """
    N = 20 # number of agents
    degree = 4 # degree of the graph
    vertex_dim = 5 # vertex stalk dimension
    edge_dim = 5 # edge stalk dimension
    sheaf_type = "constant" # choices: "bundle", "weighted", "constant"

    """
    Generate sheaf and normalized sheaf Laplacian
    """
    graph_rng = MersenneTwister(seeds.graph_seed)
    g = random_regular_graph(N, degree; rng=graph_rng) # random graph

    sheaf_rng = MersenneTwister(seeds.sheaf_seed)
    s = sheaf_from_graph(sheaf_rng, g, vertex_dim, edge_dim; style=sheaf_type) # sheaf
    
    L = sheaf_laplacian_matrix(s) # sheaf Laplacian
    L = block_normalize_matrix(L, N, vertex_dim)
    K = opnorm(L, 2) # spectral radius
    η = smallest_nonzero_eigenvalue(L)

    """
    Experiment parameters
    """
    init_cond_rng = MersenneTwister(seeds.init_cond_seed)
    x0 = randn(init_cond_rng, vertex_dim * N) # initial condition

    B_min = 20 # minimum delay
    B_max = 40 # maximum delay
    γ = 0.01 # step-size
    T = 10000 # number of iterations

    println("Running experiment with seeds: ", seeds)
    println("num_agents: ", N)
    println("step_size: ", γ)
    println("spectral_radius: ", K)
    println("frustration: ", η)
    println("B_max: ", B_max)

    delay_rng_sync = MersenneTwister(seeds.delay_seed)
    delay_rng_async = MersenneTwister(seeds.delay_seed)

    alpha_sync, _ = compute_trajectory(delay_rng_sync, L, x0, γ, N, vertex_dim; max_iters=T)
    alpha_async, _ = compute_trajectory(delay_rng_async, L, x0, γ, N, vertex_dim; B_min=B_min, B_max=B_max, max_iters=T)

    p = plot(yscale=:log10, title="Sync v.s. Async", xlabel="t", ylabel="Q(x(t))") # initialize the plot
    plot!(p, alpha_sync, label="sync", color=:blue, alpha=0.75)
    plot!(p, alpha_async, label="async", color=:orange, alpha=0.75)

    display(p)
end

"""
run the experiment
"""
seeds = generate_seeds()
run_experiment(seeds)
