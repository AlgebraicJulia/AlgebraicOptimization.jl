module HomologicalPrograms

export MultiAgentMPCProblem, MPCParams, ADMM, solve, do_mpc!, NonConvexADMM, HomologicalProgram, AbstractHomologicalProgram,
    CollocationHP

using BlockArrays
using LinearAlgebra
using JuMP
using Optim
using ..CellularSheaves
using ..MPC


abstract type OptimizationAlgorithm end

struct ADMM <: OptimizationAlgorithm
    step_size::Real
    num_iters::Int
end

struct NonConvexADMM <: OptimizationAlgorithm
    step_size::Real
    num_iters::Int
    gd_step_size::Real
    gd_num_iters::Int
end


abstract type AbstractHomologicalProgram end

# Stuff for generic homological programs

struct HomologicalProgram <: AbstractHomologicalProgram
    objectives::Vector{Function} # X -> R
    sheaf::AbstractCellularSheaf
    b::AbstractArray
end

struct CollocationHP <: AbstractHomologicalProgram
    node_solvers::Vector{Function} # X -> X
    sheaf::AbstractCellularSheaf
end


function solve(h::CollocationHP, alg::ADMM)
    y = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    z = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    x_star = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)

    #regularized_objectives = [(z, y) -> (x -> f(x) + alg.step_size / 2 * (x - z + y)' * (x - z + y)) for f in h.objectives]

    for k in 1:alg.num_iters
        for (i, f) in enumerate(h.node_solvers)
            res_x = f(z - y)
            #res_x = optimize(f(z[Block(i)], y[Block(i)]), zeros(h.sheaf.vertex_stalks[i]), LBFGS(); autodiff=:forward)
            println(length(x_star[Block(i)]))
            println(length(res_x))
            x_star[Block(i)] = res_x
        end
        z = nearest_section(h.sheaf, x_star + y)

        y = y + x_star - z
    end
    return z, y
end


function solve(h::HomologicalProgram, alg::ADMM)
    y = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    z = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    x_star = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)

    regularized_objectives = [(z, y) -> (x -> f(x) + alg.step_size / 2 * (x - z + y)' * (x - z + y)) for f in h.objectives]

    for k in 1:alg.num_iters
        for (i, f) in enumerate(regularized_objectives)
            res_x = optimize(f(z[Block(i)], y[Block(i)]), zeros(h.sheaf.vertex_stalks[i]), LBFGS(); autodiff=:forward)
            x_star[Block(i)] = Optim.minimizer(res_x)
        end
        z = nearest_section(h.sheaf, x_star + y, h.b)

        y = y + x_star - z
    end
    return z, y
end

# Stuff for MPC based homological programs

struct MPCParams
    Q::Matrix
    R::Matrix
    ls::DiscreteLinearSystem
    control_bounds
    horizon
    x_target::AbstractArray
end

MPCParams(Q, R, ls, cbs, N) = MPCParams(Q, R, ls, cbs, N, zeros(size(Q)[1]))


struct NonLinearHomologicalProgram <: AbstractHomologicalProgram
    objectives::Vector{JuMP.Model}
    sheaf::PotentialSheaf
end

struct MultiAgentMPCProblem <: AbstractHomologicalProgram
    objectives::Vector{MPCParams}
    sheaf::AbstractCellularSheaf
    x_curr::BlockArray
    b::AbstractArray
end

MultiAgentMPCProblem(objectives::Vector{MPCParams}, sheaf::AbstractCellularSheaf, x_curr::BlockArray) =
    MultiAgentMPCProblem(
        objectives,
        sheaf,
        x_curr,
        zeros(sum(sheaf.edge_stalks)))



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
            model = nothing
            if iszero(params.x_target)
                model = lqr_model(params.Q, params.R, params.ls, h.x_curr[Block(i)], z[Block(i)] - λ[Block(i)], params.horizon, params.control_bounds, alg.step_size)
            else
                model = lq_tracking_model(
                    params.Q, params.R, params.ls,
                    h.x_curr[Block(i)],
                    params.x_target,
                    z[Block(i)] - λ[Block(i)],
                    params.horizon, params.control_bounds, alg.step_size
                )
            end
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
        z = nearest_section(h.sheaf, x_star + λ, h.b)

        # dual update
        λ = λ + x_star - z
    end

    return u_star
end

function solve(h::MultiAgentMPCProblem, alg::NonConvexADMM) # assumes PotentialSheaves
    # Storage for optimal final state, control input, and dual variables
    λ = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    z = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    x_star = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    u_dims = [size(p.R)[2] for p in h.objectives]
    u_star = BlockArray(zeros(sum(u_dims)), u_dims)

    for k in 1:alg.num_iters
        for (i, params) in enumerate(h.objectives)
            # Contruct the optimization model
            model = nothing
            if iszero(params.x_target)
                model = lqr_model(params.Q, params.R, params.ls, h.x_curr[Block(i)], z[Block(i)] - λ[Block(i)], params.horizon, params.control_bounds, alg.step_size)
            else
                model = lq_tracking_model(
                    params.Q, params.R, params.ls,
                    h.x_curr[Block(i)],
                    params.x_target,
                    z[Block(i)] - λ[Block(i)],
                    params.horizon, params.control_bounds, alg.step_size
                )
            end
            set_silent(model)
            optimize!(model)

            # optimize each node objective

            #x_j_star, u_star = optimize_step(h.x_curr[Block(j)], o.Q, o.R, o.ls, λ[Block(j)], alg.step_size)
            x_star[Block(i)] = value.(model[:x][:, params.horizon])

            if k == alg.num_iters
                u_star[Block(i)] = value.(model[:u][:, 1])
            end
        end

        # Minimize the gradient of the subproblem
        z = x_star + λ

        objective(x) = potential_objective(h.sheaf)(x) + norm(x_star - x + λ)
        res = optimize(objective, z, LBFGS(); autodiff=:forward)
        z = Optim.minimizer(res)
        #=
        for i in 1:alg.gd_num_iters
            update = apply_Laplacian(h.sheaf, z) + alg.step_size * (x_star + λ)
            if norm(update) < 0.01
                println("Solved within tolerance!")
                break
            end
            z -= alg.gd_step_size * update
        end
        # project results onto a global section
        #z = nearest_section(h.sheaf, x_star + λ, h.b)=#

        # dual update
        λ = λ + x_star - z
    end

    return u_star
end

function mpc_step!(h::MultiAgentMPCProblem, alg::OptimizationAlgorithm)
    u_star = solve(h, alg)


    for (i, p) in enumerate(h.objectives)
        res = p.ls(h.x_curr[Block(i)], u_star[Block(i)])
        h.x_curr[Block(i)] = res
    end


    return u_star
end

function do_mpc!(h::MultiAgentMPCProblem, alg::OptimizationAlgorithm, nsteps::Int)
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