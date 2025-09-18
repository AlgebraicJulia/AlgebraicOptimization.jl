using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays
using Graphs
using Distributions

Random.seed!(69)

# Build a sheaf with a circle topology
n_agents = 2
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

#rm_generator(stalk_dim) = rand(2, stalk_dim)
rm_generator(stalk_dim) = Matrix{Float64}(I(stalk_dim))

#g = random_regular_graph(n_agents, 4)
g = path_graph(n_agents)
#g = erdos_renyi(n_agents, 0.2)
#g = complete_graph(n_agents)
#g = wheel_graph(n_agents)
#println(is_connected(g))

s = sheaf_from_graph(g, 4, rm_generator)

L = Array(sparse(sheaf_laplacian_matrix(s)))

# Test standard synchronous gradient descent convergence with theoretically optimal stepsize.
K = opnorm(L, 2)




energy_function(L) = x -> 0.5 * x' * (L * x)

function compute_trajectory(L, x0, γ; max_iters=1000, tol=1e-8)
    f = energy_function(L)
    losses = [f(x0)]
    x_curr = x0
    for i in 1:max_iters
        x_curr = x_curr - γ * (L * x_curr)

        if i % (0.1 * max_iters) == 0
            push!(losses, f(x_curr))
        end
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
    d1 = Normal(0.1 * B, 0.01 * B)
    d2 = Normal(0.8 * B, 0.05 * B)

    mixture = MixtureModel([d1, d2], [0.0, 1.0])

    #periods = rand(mixture, nblocks)
    #periods = ceil.(Int, periods)
    #println(periods)
    #phases = [rand(0:periods[i]-1) for i in 1:nblocks]
    #broadcast_probs = rand(0:0.001:0.3, nblocks)
    #update_probs = rand(nblocks)

    # Dumb test case
    periods = [B, B]
    phases = [0, 0]

    losses = [f(x0)]

    for t in 1:max_iters
        # Test that local states are stale i.e. columns have different values
        #=if t % 1000 == 0
            stale = true
            for i in 1:nblocks
                stale &= norm(local_states[:, Block(i)] - global_state) > 1e-4
            end
            println(stale)
        end=#

        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], nblocks), ones(Int, nblocks))
        for i in 1:nblocks
            #if rand() < update_probs[i]
            # update local state
            local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
            # update global state
            global_state[Block(i)] = local_states[Block(i), Block(i)][:]
            #end
            if t % periods[i] == phases[i]
                #if rand() < broadcast_probs[i]
                # if it's the right time, broadcast your local state to other agents
                x = local_states[Block(i), Block(i)]
                local_states[Block(i), :] .= x
                #=for j in 1:nblocks
                    local_states[Block(i), Block(j)] = x 
                end=#
            end
        end
        #println(local_states)
        if t % (0.1 * max_iters) == 0
            push!(losses, f(global_state))
        end
        #=if losses[end] < tol
            break
        end=#
    end
    return losses, global_state
end


T = Int(5e3)
B = 30

γ_synch = 1 / K
#γ_synch = 1
γ_asynch = 2 / (K * (1 + 2 * (sqrt(n_agents) * B)))

x0 = vcat(repeat([5.0], 4)..., repeat([15.0], 4)...)

energies_synch, final_state_synch = compute_trajectory_asynch(L, x0, γ_synch, n_agents, 4; B=B, max_iters=T)
energies_asynch, final_state_asynch = compute_trajectory_asynch(L, x0, γ_asynch, n_agents, 4; B=B, max_iters=T)

plt = plot(energies_synch, title="Synch SS vs Asynch SS", xlabel="t", ylabel="Energy", label="1/K")
plot!(plt, energies_asynch, label="2/(K(1+2√(n)B))")

function run_experiments(T)
    plts = []
    for i in 1:20

        x0 = rand(-5:0.01:5, 4 * n_agents)

        energies_synch, final_state_synch = compute_trajectory(L, x0, γ_synch; max_iters=T)
        energies_asynch, final_state_asynch = compute_trajectory_asynch(L, x0, γ_synch, n_agents, 4; B=B, max_iters=T)

        plt = plot(energies_synch, yscale=:log10, title="Synch vs Asynch", xlabel="t", ylabel="Energy", label="Synch")
        plot!(plt, energies_asynch, yscale=:log10, label="Asynch")
        push!(plts, plt)
    end
    return plts
end
plt
#plts = run_experiments(T);
