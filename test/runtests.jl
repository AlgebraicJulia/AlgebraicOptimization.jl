using Test

using AlgebraicOptimization

@testset "Open Optimization Problems" begin
  include("Objectives.jl")
end

@testset "Open Optimizers" begin
  include("Optimizers.jl")
end

  @testset "Open Flow Graphs" begin
    include("OpenFlowGraphs.jl")
  end


