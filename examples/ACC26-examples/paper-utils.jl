using AlgebraicOptimization
using BlockArrays
using Plots
gr()
default(fontfamily="Computer Modern")
using CSV
using Tables

rgb(r, g, b) = RGB(r / 255.0, g / 255.0, b / 255.0)

const blue = rgb(97, 136, 178)
const orange = rgb(223, 167, 119)
const green = rgb(172, 207, 146)
#const purple = rgb(216, 201, 238)
const purple = rgb(216, 150, 238)
#const beige = rgb(250, 238, 203)
const beige = rgb(250, 200, 203)

colors = [blue, orange, green, purple, beige]


function random_psd(dim)
    A = rand(ceil(Int64, dim / 2), dim)
    return A' * A
end

function random_pd(dim)
    A = rand(ceil(Int64, dim / 2), dim)
    return A' * A + I(dim)
end

function matrix_weighted_rm(A)
    _, U = qr(A)
    return U
end

function matrix_weighted_edge_generator(; pd_prob=0.5)
    return stalk_dim -> begin
        if rand() < pd_prob
            A = random_pd(stalk_dim)
            return matrix_weighted_rm(A)
        else
            A = random_psd(stalk_dim)
            return matrix_weighted_rm(A)
        end
    end
end


function random_semi_orthogonal_matrix(n, m)
    N = max(n, m)
    A = rand(N, N)
    Q, _ = qr(A)
    return Q[1:n, 1:m]
end

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
    dists::Vector{Normal}
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
    # clamp to be in the range of 1:B
    update_periods = [p > B ? B : p for p in update_periods]
    update_periods = [p < 1 ? 0 : p for p in update_periods]
    update_phases = [rand(0:update_periods[i]-1) for i in 1:nblocks]

    println(update_periods)
    println(update_phases)

    broadcast_mixture = MixtureModel(broadcast_model.dists, broadcast_model.weights)

    broadcast_periods = ceil.(Int, rand(broadcast_mixture, nblocks))
    # clamp to be in the range of 1:B
    broadcast_periods = [p > B ? B : p for p in broadcast_periods]
    broadcast_periods = [p < 1 ? 0 : p for p in broadcast_periods]
    broadcast_phases = [rand(0:broadcast_periods[i]-1) for i in 1:nblocks]

    println(broadcast_periods)
    println(broadcast_phases)

    traj = [x0]

    for t in 1:max_iters
        # every agent computes a local update
        g = BlockArray(L * local_states, repeat([block_size], nblocks), ones(Int, nblocks))
        for i in 1:nblocks
            if t % update_periods[i] == update_phases[i]
                # update local state
                local_states[Block(i), Block(i)] -= γ * g[Block(i), Block(i)]
                #local_states[Block(i), Block(i)] ./= norm(local_states[Block(i), Block(i)])
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

function save_trajectory(filename, trajectory)
    CSV.write(filename, Tables.table(trajectory))
end

function save_trajectories(path, experiment_name, trajectories)
    for (i, t) in enumerate(trajectories)
        f = path * experiment_name * "_traj$(i).csv"
        save_trajectory(f, t)
    end
end

function load_trajectory(trajectory_file)
    return CSV.File(trajectory_file) |> CSV.Tables.matrix
end

function empty_experiment_plot(x_label, y_label; kwargs...)
    plt = plot(yformatter=:plain, xformatter=:plain; kwargs...)
    plot!(plt, title="", xlabel=x_label, ylabel=y_label, thickness_scaling=1.5)
    return plt
end

function plot_log_loss_curve!(plt, losses, label; kwargs...)
    plot!(plt, yscale=:log10, losses, label=label, linewidth=2; kwargs...) #=color=color,=#
end

function plot_loss_curve!(plt, losses, label; kwargs...)
    plot!(plt, losses, label=label, linewidth=2; kwargs...) #=color=color,=#
end

