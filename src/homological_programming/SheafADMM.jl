module SheafADMM

using ..CellularSheaves
import ..CellularSheaves: optimize!

struct ADMM <: OptimizationAlgorithm
    step_size::Float64
    max_iters::Int
    epsilon::Float64
end

function optimize!(s::SheafObjective, alg::ADMM)
    

end

end