module HomologicalPrograms

export HomologicalProgram, solve

using ..NetworkSheaves
using ProximalOperators, ProximalAlgorithms
using SparseArrays

struct HomologicalProgram{T<:AbstractNetworkSheaf}
    # Node objectives must be callable and implement the prox interface
    # defined in ProximalOperators.jl
    node_objectives::Vector
    sheaf::T
    # Use this matrix if only some decision variables are involved in the
    # sheaf objective term. This should be a projection matrix:
    # oplus_{i=1}^{length(node_objectives)} R^{k_i} -> C^0(sheaf)
    # where k_i is the dimension of the i-th node objective.
    #P::Union{Matrix{Float64},Nothing}
end

function solve(prob::HomologicalProgram{EuclideanSheaf{Float64}}, alg::T=ProximalAlgorithms.DouglasRachford()) where T<:ProximalAlgorithms.IterativeAlgorithm
    #if typeof(alg) != ProximalAlgorithms.DouglasRachford || typeof(alg) != DRLS
    #    error("Unsupported algorithm for HomologicalProgram with EuclideanSheaf. Should be DouglasRachford or DRLS.")
    #end

    s = prob.sheaf
    N = sum(vertex_stalks(s))

    objs = Tuple(prob.node_objectives)
    # Build slices to index decision variables for each node objective
    start_idx = 1
    slices = []
    for n in vertex_stalks(s)
        end_idx = start_idx + n - 1
        push!(slices, (start_idx:end_idx,))
        start_idx = end_idx + 1
    end

    f = SlicedSeparableSum(objs, Tuple(slices))
    g = IndAffine(sparse(coboundary_map(s)), zeros(size(coboundary_map(s), 1)); iterative=true)

    return alg(f=f, g=g, x0=zeros(N), gamma=1.0)
end




end

