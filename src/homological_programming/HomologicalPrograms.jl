module HomologicalPrograms

export MultiAgentMPCProblem, MPCParams, ADMM, solve

using BlockArrays
using ..CellularSheaves
using ..MPC

abstract type HomologicalProgam end

struct MPCParams
    Q::Matrix
    R::Matrix
    ls::DiscreteLinearSystem
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
    #λ = zeros(length(h.x_curr))
    λ = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), vertex_stalks)

    #λ = [zeros(dim) for dim in h.sheaf.vertex_stalks]

    for i in 1:alg.num_iters
        x_star = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), vertex_stalks)
        for (j, o) in enumerate(h.objectives)
            # optimize each node objective
            x_j_star, u_star = optimize_step(h.x_curr[Block(j)], o.Q, o.R, o.ls, λ[Block(j)], alg.step_size)
            x_star[Block(j)] = x_j_star
        end

        # project results onto a global section
        z = nearest_section(h.sheaf, x_star + λ)

        # dual update
        λ = λ + x_star - z
    end
end

# objective_function(model) gets obj function

end