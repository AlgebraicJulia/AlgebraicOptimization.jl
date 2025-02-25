module AlgebraicOptimization

using Reexport

include("compositional_programming/CompositionalProgramming.jl")
include("homological_programming/HomologicalProgramming.jl")

@reexport using .CompositionalProgramming
@reexport using .HomologicalProgramming

end
