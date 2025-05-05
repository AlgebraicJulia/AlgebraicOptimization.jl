module CellularSheaves

export AbstractCellularSheaf, CellularSheaf, PotentialSheaf, nearest_section, set_edge_maps!, Laplacian, apply_Laplacian, coboundary_map, apply_coboundary_map,
    potential_objective

using BlockArrays
using SparseArrays
using LinearOperators
using Krylov
using LinearAlgebra
using Graphs
using ForwardDiff
using MLStyle: @match 

abstract type AbstractCellularSheaf end

struct CellularSheaf <: AbstractCellularSheaf
    vertex_stalks::Vector{Int}
    edge_stalks::Vector{Int}
    coboundary::BlockArray
end

function CellularSheaf(vertex_stalks::Vector{Int}, edge_stalks::Vector{Int})
    cb = BlockArray(spzeros(sum(edge_stalks), sum(vertex_stalks)), edge_stalks, vertex_stalks)

    return CellularSheaf(vertex_stalks, edge_stalks, cb)
end

struct PotentialSheaf <: AbstractCellularSheaf
    vertex_stalks::Vector{Int}
    edge_stalks::Vector{Int}
    coboundary::BlockArray
    potentials::Vector{Function}
end

function PotentialSheaf(vertex_stalks::Vector{Int}, edge_stalks::Vector{Int}, potentials)
    cb = BlockArray(spzeros(sum(edge_stalks), sum(vertex_stalks)), edge_stalks, vertex_stalks)

    return PotentialSheaf(vertex_stalks, edge_stalks, cb, potentials)
end

function potential_objective(s::PotentialSheaf)
    total_potential(y) = sum([potential(y[Block(e)]) for (e, potential) in enumerate(s.potentials)])
    return x -> total_potential(s.coboundary * x)
end

#=
function constant_sheaf(g::Graph, dimension::Int)
    s = CellularSheaf(repeat([dimension], nv(g)), repeat([dimension], ne(g)))
    for 
end=#

function set_edge_maps!(s::AbstractCellularSheaf, v1::Int, v2::Int, e::Int, rm1::AbstractMatrix, rm2::AbstractMatrix)
    @assert size(rm1) == (s.edge_stalks[e], s.vertex_stalks[v1])
    @assert size(rm2) == (s.edge_stalks[e], s.vertex_stalks[v2])
    s.coboundary[Block(e), Block(v1)] = rm1
    s.coboundary[Block(e), Block(v2)] = -rm2
end

"""     nearest_section(s::CellularSheaf, x)

Computes the projection of x onto the space of global sections of s.
"""
function nearest_section(s::CellularSheaf, x)
    # Compute the projection of x onto the space of global sections of s
    d = coboundary_map(s)

    eL = LinearOperator(d) * LinearOperator(d')

    b = d * x

    y, stats = cg(eL, Array(b))
    #println(stats)

    return BlockArray(x - d' * y, s.vertex_stalks)
end


"""     nearest_section(s::CellularSheaf, x, b)

Computes the projection of x onto the subspace satisfying
    δx = b
where δ is the coboundary map of s.
"""
function nearest_section(s::CellularSheaf, x, b)
    d = coboundary_map(s)

    eL = LinearOperator(d) * LinearOperator(d')

    rhs = d * x - b

    y, stats = cg(eL, Array(rhs))
    #println(stats)

    return BlockArray(x - d' * y, s.vertex_stalks)

end

function Laplacian(s::CellularSheaf)
    return s.coboundary' * s.coboundary
end

function apply_Laplacian(s::PotentialSheaf, x)
    total_potential(y) = sum([potential(y[Block(e)]) for (e, potential) in enumerate(s.potentials)])
    d = s.coboundary
    return d' * ForwardDiff.gradient(total_potential, d * x)
end

function Laplacian(s::PotentialSheaf)

end

function apply_Laplacian(s::CellularSheaf, x)
    return s.coboundary' * s.coboundary * x
end

function coboundary_map(s::CellularSheaf)
    return s.coboundary
end

function apply_coboundary_map(s::CellularSheaf, x)
    return s.coboundary * x
end





struct ThreadedSheaf <: AbstractCellularSheaf


end

end 