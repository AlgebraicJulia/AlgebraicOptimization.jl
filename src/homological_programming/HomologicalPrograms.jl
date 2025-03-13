module HomologicalPrograms

export MultiAgentMPCProblem, MPCParams, ADMM, solve, do_mpc!

using BlockArrays
using JuMP
using ..CellularSheaves
using ..MPC

abstract type HomologicalProgam end

struct MPCParams
    Q::Matrix
    R::Matrix
    ls::DiscreteLinearSystem
    control_bounds
    horizon
end

struct MultiAgentMPCProblem <: HomologicalProgam
    objectives::Vector{MPCParams}
    sheaf::AbstractCellularSheaf
    x_curr::BlockArray
end

abstract type OptimizationAlgorithm end

struct ADMM <: OptimizationAlgorithm
    step_size::Real
    num_iters::Int
end

function solve(h::MultiAgentMPCProblem, alg::ADMM)
    # Storage for optimal final state, control input, and dual variables
    λ = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    z = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    x_star = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    u_dims = [size(p.R)[2] for p in h.objectives]
    u_star = BlockArray(zeros(sum(u_dims)), u_dims)

    for k in 1:alg.num_iters
        for (i, params) in enumerate(h.objectives)
            # Contruct the optimization model
            model = lqr_model(params.Q, params.R, params.ls, h.x_curr[Block(i)], z[Block(i)] - λ[Block(i)], params.horizon, params.control_bounds, alg.step_size)
            set_silent(model)
            optimize!(model)

            # optimize each node objective

            #x_j_star, u_star = optimize_step(h.x_curr[Block(j)], o.Q, o.R, o.ls, λ[Block(j)], alg.step_size)
            x_star[Block(i)] = value.(model[:x][:, params.horizon])

            if k == alg.num_iters
                u_star[Block(i)] = value.(model[:u][:, 1])
            end
        end

        # project results onto a global section
        z = nearest_section(h.sheaf, x_star + λ)

        # dual update
        λ = λ + x_star - z
    end

    return u_star
end

function mpc_step!(h::MultiAgentMPCProblem, alg::ADMM)
    u_star = solve(h, alg)


    for (i, p) in enumerate(h.objectives)
        res = p.ls(h.x_curr[Block(i)], u_star[Block(i)])
        h.x_curr[Block(i)] = res
    end


    return u_star
end

function do_mpc!(h::MultiAgentMPCProblem, alg::ADMM, nsteps::Int)
    trajectory = []
    controls = []
    push!(trajectory, deepcopy(h.x_curr))


    for i in 1:nsteps
        u = mpc_step!(h, alg)
        push!(controls, u)
        push!(trajectory, deepcopy(h.x_curr))

    end
    return trajectory, controls
end

end