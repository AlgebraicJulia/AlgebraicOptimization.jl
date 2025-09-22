using AlgebraicOptimization
using BlockArrays
using Plots
using CSV
using Tables


# Trajectory Computation Utils

function compute_trajectory(L, x0, γ; max_iters=1000, tol=1e-8)
    f = energy_function(L)
    traj = [x0]
    x_curr = x0
    for i in 1:max_iters
        x_curr = x_curr - γ * (L * x_curr)

        push!(traj, x_curr)

        if f(x_curr) < tol
            break
        end
    end
    return traj
end

# Pass this for an asynch sim with random periods and phases drawn from mixture model
struct MixtureModelParams
    dists::Vector{Distribution}
    weights::Vector{Float64}
end

# Pass this for an asynch sim with update and broadcast coin flips
struct ProbabilisticModelParams
    update_prob_upper_bound::Float64
    broadcast_prob_upper_bound::Float64
end

function compute_trajectory_asynch(L, x0, γ, nblocks, block_size, params::ProbabilisticModelParams; max_iters=1000, tol=1e-8, B=50)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], nblocks))
    local_states = BlockArray(hcat([global_state for _ in 1:nblocks]...), repeat([block_size], nblocks), ones(Int, nblocks))

    broadcast_probs = rand(0.0:(0.01*params.broadcast_prob_upper_bound):params.broadcast_prob_upper_bound, nblocks)
    update_probs = rand(0.0:(0.01*params.update_prob_upper_bound):params.update_prob_upper_bound, nblocks)

    traj = [x0]

    for t in 1:max_iters
        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], nblocks), ones(Int, nblocks))
        for i in 1:nblocks
            if rand() < update_probs[i]
                # update local state
                local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
                # update global state
                global_state[Block(i)] = local_states[Block(i), Block(i)][:]
            end
            if rand() < broadcast_probs[i]
                x = local_states[Block(i), Block(i)]
                local_states[Block(i), :] .= x
            end
        end
        push!(traj, global_state)

        if f(traj[end]) < tol
            break
        end
    end
    return traj
end

function compute_trajectory_asynch(L, x0, γ, nblocks, block_size, update_model::MixtureModelParams, broadcast_model::MixtureModelParams; max_iters=1000, tol=1e-8, B=50)
    f = energy_function(L)
    global_state = BlockArray(x0, repeat([block_size], nblocks))
    local_states = BlockArray(hcat([global_state for _ in 1:nblocks]...), repeat([block_size], nblocks), ones(Int, nblocks))

    update_mixture = MixtureModel(update_model.dists, update_model.weights)

    update_periods = ceil.(Int, rand(update_mixture, nblocks))
    update_phases = [rand(0:update_periods[i]-1) for i in 1:nblocks]

    broadcast_mixture = MixtureModel(broadcast_model.dists, broadcast_model.weights)

    broadcast_periods = ceil.(Int, rand(broadcast_mixture, nblocks))
    broadcast_phases = [rand(0:broadcast_periods[i]-1) for i in 1:nblocks]

    traj = [x0]

    for t in 1:max_iters
        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], nblocks), ones(Int, nblocks))
        for i in 1:nblocks
            if t % update_periods[i] == update_phases[i]
                # update local state
                local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
                # update global state
                global_state[Block(i)] = local_states[Block(i), Block(i)][:]
                # Resample your phase
                update_phases[i] = rand(0:update_periods[i]-1)
            end
            # if it's the right time, broadcast your local state to other agents
            if t % broadcast_periods[i] == broadcast_phases[i]
                x = local_states[Block(i), Block(i)]
                local_states[Block(i), :] .= x
                # Resample your phase
                broadcast_phases[i] = rand(0:broadcast_periods[i]-1)
            end
        end
        push!(traj, global_state)

        if f(traj[end]) < tol
            break
        end
    end
    return traj
end

# Plotting Utils

function save_trajectory_csv(traj, filename::String)

end
