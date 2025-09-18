using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays
using Graphs
using Distributions

Random.seed!(21)

function random_orthogonal_matrix(dim::Int)
    A = randn(dim, dim)
    return Matrix(qr(A).Q)
end

function sheaf_from_graph(g::Graph, vertex_dim::Int, edge_dim::Int, rm_generator::Function)
    """
    Sheaf generator
    """
    s = EuclideanSheaf{Float64}(repeat([vertex_dim], nv(g)))

    for e in edges(g)
        i, j = src(e), dst(e)
        rm1 = rm_generator(vertex_dim,edge_dim)
        rm2 = rm_generator(vertex_dim,edge_dim)
        add_sheaf_edge!(s, i, j, rm1, rm2)
    end
    return s
end

energy_function(L) = x -> 0.5 * x' * (L * x) # Dirichlet energy function


function compute_trajectory(L, x0, γ, num_blocks, block_size; B_min=1,B_max=1, max_iters=1000, tol=1e-8)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], num_blocks))
    local_states = BlockArray(hcat([global_state for _ in 1:num_blocks]...), repeat([block_size], num_blocks), ones(Int, num_blocks))

    periods = rand(B_min:B_max,num_blocks) # set the communication periods, i.e. delays
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

# Sheaf parameters
N = 20 # number of agents
degree = 4 # degree of the graph
vertex_dim = 4 # vertex stalk dimension
edge_dim = 4 # edge stalk dimension
rm_generator(vertex_dim,edge_dim) = rand(edge_dim, vertex_dim) # Restriction map generator
# rm_generator(vertex_dim,edge_dim) = random_orthogonal_matrix(vertex_dim)

# Generate sheaf
g = random_regular_graph(N, degree; seed=69) # random graph
s = sheaf_from_graph(g, vertex_dim,edge_dim,rm_generator) # sheaf
L = Array(sparse(sheaf_laplacian_matrix(s))) # sheaf Laplacian

# Experiment parameters
x0 = rand(vertex_dim * N) # initial condition
B_min = 10 # minimum delay
B_max = 10 # maximum delay
γ = 0.03
T = 5000 # number of iterations

# Theoretical step-sizes that will converge
K = opnorm(L, 2)
γ_synch = 1 / K
γ_asynch = 1 / (K * (1 + 2 * (sqrt(N) * B_max)))

# energies_synch, final_state_synch = compute_trajectory(L, x0, γ; max_iters=T)
energies_synch, final_state_synch = compute_trajectory(L, x0, γ, N, vertex_dim; max_iters=T)
energies_asynch, final_state_asynch = compute_trajectory(L, x0, γ, N, vertex_dim; B_min=B_min, B_max=B_max, max_iters=T)

plot(energies_synch, yscale=:log10, title="Sync v.s. Async", xlabel="t", ylabel="Q(x(t))", label="Sync")
plot!(energies_asynch, yscale=:log10, label="Async")