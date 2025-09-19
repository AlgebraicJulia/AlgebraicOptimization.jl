using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays
using Graphs
using Distributions

function random_orthogonal_matrix(dim::Int)
    A = randn(dim, dim)
    return Matrix(qr(A).Q)
end

function random_psd_matrix(dim::Int)
    B = randn(dim, dim) 
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
    
    # Calculate all eigenvalues
    eigenvalues = eigvals(Symmetric(M))
    nonzero_eigenvalues = filter(e -> abs(e) > tol, eigenvalues)
    if isempty(nonzero_eigenvalues)
        return nothing
    else
        return minimum(abs.(nonzero_eigenvalues))
    end
end

"""
Types of restriction maps
"""
rm_bundle(vertex_dim::Int, edge_dim::Int) = Matrix{Float64}(randn() * random_orthogonal_matrix(vertex_dim))
rm_random(vertex_dim::Int, edge_dim::Int) = Matrix{Float64}(randn(edge_dim, vertex_dim))
rm_identity(vertex_dim::Int, edge_dim::Int) = Matrix{Float64}(I(vertex_dim))
rm_weighted(vertex_dim::Int, edge_dim::Int) = Matrix{Float64}(random_psd_matrix(vertex_dim))

function sheaf_from_graph(g::Graph, vertex_dim::Int, edge_dim::Int; style::String=nothing)
    """
    Sheaf generator
    """
    s = EuclideanSheaf{Float64}(repeat([vertex_dim], nv(g)))

    for e in edges(g)
        i, j = src(e), dst(e)
        if style == "weighted"
            if vertex_dim != edge_dim
                error("Type Error: vertex dimension must equal edge_dim for a matrix-weighted graph")
            else
                W = rm_weighted(vertex_dim, edge_dim)
                rm1 = W
                rm2 = W
            end
        elseif style == "bundle"
            if vertex_dim != edge_dim
                error("Type Error: vertex dimension must equal edge_dim for a discrete vector bundle")
            else
                rm1 = rm_bundle(vertex_dim,edge_dim)
                rm2 = rm_identity(vertex_dim,edge_dim)
            end
        elseif style == "constant"
            if vertex_dim != edge_dim
                error("Type Error: vertex dimension must equal edge_dim for a constant sheaf")
            else
                rm1 = rm_identity(vertex_dim,edge_dim)
                rm2 = rm_identity(vertex_dim,edge_dim)
            end
        else
            rm1 = rm_random(vertex_dim, edge_dim)
            rm2 = rm_random(vertex_dim, edge_dim)
        
        end
        add_sheaf_edge!(s, i, j, rm1, rm2)
    end
    return s
end

energy_function(L) = x -> 0.5 * x' * (L * x) # Dirichlet energy function

function compute_trajectory(L, x0, γ, num_blocks, block_size; B_min=1, B_max=1, max_iters=1000, tol=1e-8)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], num_blocks))
    local_states = BlockArray(hcat([global_state for _ in 1:num_blocks]...), repeat([block_size], num_blocks), ones(Int, num_blocks))

    periods = rand(B_min:B_max, num_blocks) # set the communication periods, i.e. delays
    phases = [rand(0:periods[i]-1) for i in 1:num_blocks] # set the communication phases
    losses = [f(x0)] # initialize the losses

    for t in 1:max_iters
        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], num_blocks), ones(Int, num_blocks))
        for i in 1:num_blocks
            if t % periods[i] == phases[i]
                # update local state
                local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
                # update global state
                global_state[Block(i)] = local_states[Block(i), Block(i)][:]
                # if it's the right time, broadcast your local state to other agents
                x = local_states[Block(i), Block(i)]
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
Sheaf parameters
"""
N = 4 # number of agents
degree = 2 # degree of the graph
vertex_dim = 3 # vertex stalk dimension
edge_dim = 3 # edge stalk dimension
sheaf_type = "weighted" # choices: "bundle", "weighted", "constant"

"""
Generate sheaf and normalized sheaf Laplacian
"""

g = random_regular_graph(N, degree; seed=856) # random graph
s = sheaf_from_graph(g, vertex_dim, edge_dim; style=sheaf_type) # sheaf
L_unnormalized = sheaf_laplacian_matrix(s) # sheaf Laplacian
L =  Array(sparse(block_normalize_matrix(L_unnormalized, N, vertex_dim))) # normalized sheaf Laplacian
K = opnorm(L, 2) # spectral radius
η = smallest_nonzero_eigenvalue(L)

"""
Experiment parameters
"""
x0 = randn(vertex_dim * N) # initial condition
B_min = 20 # minimum delay
B_max = 40 # maximum delay
γ =  0.1 # step-size
T = 10000 # number of iterations

println("num_agents: ", N)
println("step_size: ", γ)
println("spectral_radius: ", K)
println("frustration: ", η)
println("B: ", B_max)

x0 = randn(vertex_dim * N) # initial condition

alpha_sync, _ = compute_trajectory(L, x0, γ, N, vertex_dim; max_iters=T)
alpha_async, _ = compute_trajectory(L, x0, γ, N, vertex_dim; B_min=B_min, B_max=B_max, max_iters=T)

p = plot(yscale=:log10, title="Sync v.s. Async", xlabel="t", ylabel="Q(x(t))") # initialize the plot
plot!(p, alpha_sync, label="sync", color=:blue, alpha=0.75)
plot!(p, alpha_async, label="async", color=:orange, alpha=0.75)

display(p)