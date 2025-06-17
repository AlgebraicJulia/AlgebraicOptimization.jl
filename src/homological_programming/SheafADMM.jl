module SheafADMM

using Optim
using ..CellularSheaves
import ..CellularSheaves: optimize!

abstract type AbstractHomologicalProgam end


struct ADMM <: OptimizationAlgorithm
    step_size::Float64
    max_iters::Int
end


struct HomologicalProgram <: AbstractHomologicalProgam
    objectives::Vector{Function}
    sheaf::AbstractCellularSheaf
end



function solve(h::HomologicalProgram, alg::ADMM)
    y = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)
    z = BlockArray(zeros(sum(h.sheaf.vertex_stalks)), h.sheaf.vertex_stalks)

    regularized_objectives = [(z, y) -> (x -> f(x) + alg.step_size / 2 * (x - z + y)' * (x - z + y)) for f in h.objectives]

    for k in 1:alg.num_iters
        for (i, f) in enumerate(regularized_objectives)
            res_x = optimize(f(z[Block(i)], y[Block(i)]), zeros(h.sheaf.vertex_stalks[i]), LBFGS(); autodiff=:forward)

        end
    end




end

end