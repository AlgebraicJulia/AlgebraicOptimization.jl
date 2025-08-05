module CellularSheaves

export AbstractCellularSheaf, CellularSheaf, PotentialSheaf, nearest_section, set_edge_maps!, Laplacian, apply_Laplacian, coboundary_map, apply_coboundary_map,
    potential_objective, is_global_section, neighbor_edges

using BlockArrays
using SparseArrays
using LinearOperators
using Krylov
using LinearAlgebra
using Graphs
import Graphs
using ForwardDiff
using MLStyle: @match

abstract type AbstractCellularSheaf end
#=
struct CellularSheaf <: AbstractCellularSheaf
    vertex_stalks::Vector{Int}
    edge_stalks::Vector{Int}
    coboundary::BlockArray
    underlying_g::Graph
end

function CellularSheaf(vertex_stalks::Vector{Int}, edge_stalks::Vector{Int})
    cb = BlockArray(spzeros(sum(edge_stalks), sum(vertex_stalks)), edge_stalks, vertex_stalks)

    return CellularSheaf(vertex_stalks, edge_stalks, cb, Graph(length(vertex_stalks)))
end=#
struct SheafEdge
    v1::Int
    v2::Int
    rm12::AbstractArray
    rm21::AbstractArray
end

struct CellularSheaf <: AbstractCellularSheaf
    vertex_stalks::Vector{Int}
    edge_stalks::Vector{Int}
    edges::Vector{SheafEdge}
    underlying_graph::Graph
    edge_map::Dict{Tuple{Int,Int},Int}
end

function CellularSheaf(v_stalks::Vector{Int})
    return CellularSheaf(
        v_stalks, Int[], SheafEdge[], Graphs.Graph(length(v_stalks)), Dict{Tuple{Int,Int},Int}()
    )
end

function neighbor_edges(s::CellularSheaf, v::Int)::Vector{SheafEdge}
    ns = neighbors(underlying_graph(s), v)
    res = SheafEdge[]

    for w in ns
        edge_idx = s.edge_map[(v, w)]
        push!(res, s.edges[edge_idx])
    end
    return res
end

function set_edge_maps!(s::CellularSheaf, v1::Int, v2::Int, rm12::AbstractMatrix, rm21::AbstractMatrix)
    #@assert size(rm12) == (s.edge_stalks[e], s.vertex_stalks[v1])
    #@assert size(rm21) == (s.edge_stalks[e], s.vertex_stalks[v2]) TODO: make this correct
    #s.coboundary[Block(e), Block(v1)] = rm1
    #s.coboundary[Block(e), Block(v2)] = -rm2
    @assert size(rm12)[1] == size(rm21)[1]
    add_edge!(underlying_graph(s), v1, v2)
    e = SheafEdge(v1, v2, rm12, rm21)
    push!(s.edges, e)
    s.edge_map[(v1, v2)] = length(s.edges)
    s.edge_map[(v2, v1)] = length(s.edges)
    push!(s.edge_stalks, size(rm12)[1])
end

function underlying_graph(s::AbstractCellularSheaf)
    return s.underlying_graph
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
    add_edge!(underlying_graph(s), v1, v2)
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

function is_global_section(s::CellularSheaf, v)
    return iszero(s.coboundary * v)      # This may only work if the graph underlying s is connected
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