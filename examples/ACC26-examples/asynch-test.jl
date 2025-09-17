using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays
using Graphs
using Distributions

Random.seed!(1)

# Build a sheaf with a circle topology
n_agents = 20
#C = [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0]
C = rand(2, 4)
s = EuclideanSheaf{Float64}(repeat([4], n_agents))

function sheaf_from_graph(g::Graph, stalk_dim::Int, rm_generator::Function)
    n = nv(g)
    s = EuclideanSheaf{Float64}(repeat([stalk_dim], n))

    for e in edges(g)
        i, j = src(e), dst(e)
        rm1 = rm_generator(stalk_dim)
        rm2 = rm_generator(stalk_dim)
        add_sheaf_edge!(s, i, j, rm1, rm2)
    end
    return s
end

#=
for i in 2:n_agents
    add_sheaf_edge!(s, i - 1, i, rand(2, 4), rand(2, 4))
end
add_sheaf_edge!(s, 1, n_agents, rand(2, 4), rand(2, 4))=#

rm_generator(stalk_dim) = rand(2, stalk_dim)

g = random_regular_graph(n_agents, 4; seed=69)
#g = path_graph(n_agents)
#g = erdos_renyi(n_agents, 0.3)
println(is_connected(g))

s = sheaf_from_graph(g, 4, rm_generator)

L = Array(sparse(sheaf_laplacian_matrix(s)))

# Test standard synchronous gradient descent convergence with theoretically optimal stepsize.
K = opnorm(L, 2)

B = 200


γ_synch = 1 / K
#γ_synch = 1
γ_asynch = 2 / (K * (1 + 2 * (sqrt(n_agents) * B)))

energy_function(L) = x -> 0.5 * x' * (L * x)

function compute_trajectory(L, x0, γ; max_iters=1000, tol=1e-8)
    f = energy_function(L)
    losses = [f(x0)]
    x_curr = x0
    for i in 1:max_iters
        x_curr = x_curr - γ * (L * x_curr)
        push!(losses, f(x_curr))
        if f(x_curr) < tol
            break
        end
    end
    return losses, x_curr
end

function compute_trajectory_asynch(L, x0, γ, nblocks, block_size; max_iters=1000, tol=1e-8, B=50)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], nblocks))
    local_states = BlockArray(hcat([global_state for _ in 1:nblocks]...), repeat([block_size], nblocks), ones(Int, nblocks))
    d1 = Normal(0.2 * B, 0.1 * B)
    d2 = Normal(0.8 * B, 0.1 * B)

    mixture = MixtureModel([d1, d2], [0.6, 0.4])

    periods = rand(mixture, nblocks)
    periods = round.(Int, periods)
    println(periods)
    phases = [rand(0:periods[i]-1) for i in 1:nblocks]
    losses = [f(x0)]

    for t in 1:max_iters
        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], nblocks), ones(Int, nblocks))
        for i in 1:nblocks
            if t % periods[i] == phases[i]
                # update local state
                local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
                # update global state
                global_state[Block(i)] = local_states[Block(i), Block(i)][:]
                # if it's the right time, broadcast your local state to other agents
                x = local_states[Block(i), Block(i)]
                local_states[Block(i), :] .= x
                #=for j in 1:nblocks
                    local_states[Block(i), Block(j)] = x 
                end=#
            end
        end
        push!(losses, f(global_state))
        if losses[end] < tol
            break
        end
    end
    return losses, global_state
end

x0 = rand(-2:0.01:2, 4 * n_agents)

T = 100000
energies_synch, final_state_synch = compute_trajectory(L, x0, γ_synch; max_iters=T)
energies_asynch, final_state_asynch = compute_trajectory_asynch(L, x0, γ_synch, n_agents, 4; B=B, max_iters=T)

plot(energies_synch, yscale=:log10, title="Synch vs Asynch", xlabel="t", ylabel="Energy", label="Synch")
plot!(energies_asynch, yscale=:log10, label="Asynch")
