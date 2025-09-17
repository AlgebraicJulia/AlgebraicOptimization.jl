using AlgebraicOptimization
using LinearAlgebra
using SparseArrays
using Plots
using Random
using BlockArrays

Random.seed!(1234)

# Build a sheaf with a circle topology
n_agents = 20
#C = [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0]
C = rand(2, 4)
s = EuclideanSheaf{Float64}(repeat([4], n_agents))

for i in 2:n_agents
    add_sheaf_edge!(s, i - 1, i, C, C)
end
add_sheaf_edge!(s, 1, n_agents, C, C)

L = Array(sparse(sheaf_laplacian_matrix(s)))

# Test standard synchronous gradient descent convergence with theoretically optimal stepsize.
K = opnorm(L, 2)

B = 100


γ_synch = 1 / K
γ_asynch = 2 / (K * (1 + 2 * (sqrt(n_agents) * B)))

energy_function(L) = x -> 0.5 * x' * (L * x)

function compute_trajectory(L, x0, γ; max_iters=10000, tol=1e-6)
    f = energy_function(L)
    traj = [x0]
    for i in 1:max_iters
        x_curr = traj[end]
        x_next = x_curr - γ * (L * x_curr)
        push!(traj, x_next)
        if f(x_next) < tol
            break
        end
    end
    return traj
end

function compute_trajectory_asynch(L, x0, γ, nblocks, block_size; max_iters=10000, tol=1e-6, B=50)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], nblocks))
    local_states = BlockArray(hcat([copy(global_state) for _ in 1:nblocks]...), repeat([block_size], nblocks), ones(Int, nblocks))
    periods = rand(1:B, nblocks)
    phases = [rand(0:periods[i]-1) for i in 1:nblocks]
    losses = [f(x0)]

    for t in 1:max_iters
        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], nblocks), ones(Int, nblocks))
        for i in 1:nblocks
            # update local state
            local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
            # update global state
            global_state[Block(i)] = vcat(local_states[Block(i), Block(i)]...)
            # if it's the right time, broadcast your local state to other agents
            if t % periods[i] == phases[i]
                x = copy(local_states[Block(i), Block(i)])
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


traj = compute_trajectory(L, x0, γ_synch);


energies_synch = [energy_function(L)(x) for x in traj]
energies_asynch, final_state_asynch = compute_trajectory_asynch(L, x0, γ_synch, n_agents, 4; B=B, max_iters=100000)

plot(energies_synch, yscale=:log10, title="Synch vs Asynch", xlabel="t", ylabel="Energy", label="Synch")
plot!(energies_asynch, yscale=:log10, label="Asynch")
