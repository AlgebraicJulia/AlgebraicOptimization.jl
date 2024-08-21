using Test
using ComponentArrays
using AlgebraicOptimization
using Random

# Term-by-term approximately equal utility functions for testing
function ≅(r1::Vector{Float64}, r2::Vector{Float64})
  if length(r1) != length(r2)
    return false
  end

  for i in eachindex(r1)
    if !isapprox(r1[i], r2[i]; rtol=.05)
      return false
    end
  end
  return true
end

function ≅(r1::Vector{Vector{Float64}}, r2::Vector{Vector{Float64}})
  if length(r1) != length(r2)
    return false
  end

  for i in eachindex(r1)
    if !(r1[i] ≅ r2[i])
      return false
    end
  end
  return true
end

function ≅(r1::ComponentArray, r2::ComponentArray)
  if length(r1) != length(r2)
    return false
  end

  for i in eachindex(r1)
    if !isapprox(r1[i], r2[i]; rtol=.05)
      return false
    end
  end
  return true
end

function seed_random()
  Random.seed!(1234)
  println("seeded 1234")
end


@testset "Open Optimization Problems" begin
  include("Objectives.jl")
end

@testset "Open Optimizers" begin
  include("Optimizers.jl")
end

  @testset "Open Flow Graphs" begin
    include("OpenFlowGraphs.jl")
  end


