# Module for network sheaves valued in Euclidean spaces with linear restriction maps
module EuclideanSheaves

export EuclideanSheaf, UnorderedPair, sheaf_laplacian_matrix, sheaf_from_graph, energy_function,
    nearest_global_section

using Graphs
using AutoHashEquals: @auto_hash_equals
using LinearOperators
using Krylov
using LinearAlgebra
using BlockArrays
import Base: hash, ==, isequal

using ..SheafInterface
import ..SheafInterface: vertex_stalks, edge_stalks, coboundary_map, add_vertex_stalk!, add_sheaf_edge!, underlying_graph,
    get_vertex_stalk, get_edge_stalk, get_restriction_map, sheaf_laplacian
using ..BlockSparseArrays

struct UnorderedPair{T}
    first::T
    second::T
end

function Base.hash(up::UnorderedPair{T}, h::UInt) where T
    return hash((up.first, up.second), h) + hash((up.second, up.first), h)
end

function Base.:(==)(up1::UnorderedPair{T}, up2::UnorderedPair{T}) where T
    return (up1.first == up2.first && up1.second == up2.second) ||
           (up1.first == up2.second && up1.second == up2.first)
end

function Base.:(isequal)(up1::UnorderedPair{T}, up2::UnorderedPair{T}) where T
    return (up1.first == up2.first && up1.second == up2.second) ||
           (up1.first == up2.second && up1.second == up2.first)
end

"""     EuclideanSheaf{T}

A Euclidean sheaf is a network sheaf where each vertex stalk is a Euclidean space R^n for some n,
each edge stalk is a Euclidean space R^m for some m, and each restriction map is a linear map
R^n -> R^m represented by a matrix of type T.
"""
@auto_hash_equals struct EuclideanSheaf{T} <: AbstractNetworkSheaf
    vertex_stalks::Vector{Int}
    edge_stalks::Dict{UnorderedPair{Int},Int}
    underlying_graph::Graph
    restriction_maps::Dict{Pair{Int},Matrix{T}}
end

EuclideanSheaf{T}(vertex_stalks::Vector{Int}) where T = EuclideanSheaf{T}(vertex_stalks, Dict{UnorderedPair{Int},Int}(), Graph(length(vertex_stalks)), Dict{Pair{Int},Matrix{T}}())


function sheaf_from_graph(g::Graph, stalk_dim::Int, rm_generator::Function; symmetric_edges=false)
    n = nv(g)
    s = EuclideanSheaf{Float64}(repeat([stalk_dim], n))

    for e in edges(g)
        i, j = src(e), dst(e)
        if symmetric_edges
            rm = rm_generator(stalk_dim)
            add_sheaf_edge!(s, i, j, rm, rm)
        else
            rm1 = rm_generator(stalk_dim)
            rm2 = rm_generator(stalk_dim)
            add_sheaf_edge!(s, i, j, rm1, rm2)
        end
    end
    return s
end

function sheaf_from_graph(g::Graph, stalk_dim::Int, rm1_generator::Function, rm2_generator::Function)
    n = nv(g)
    s = EuclideanSheaf{Float64}(repeat([stalk_dim], n))

    for e in edges(g)
        i, j = src(e), dst(e)
        rm1 = rm1_generator(stalk_dim)
        rm2 = rm2_generator(stalk_dim)
        add_sheaf_edge!(s, i, j, rm1, rm2)
    end
    return s
end

function vertex_stalks(s::EuclideanSheaf)
    return s.vertex_stalks
end

function edge_stalks(s::EuclideanSheaf)
    return s.edge_stalks
end

function underlying_graph(s::EuclideanSheaf)
    return s.underlying_graph
end

function get_vertex_stalk(s::EuclideanSheaf, v::Int)
    @assert v <= length(s.vertex_stalks)
    return s.vertex_stalks[v]
end

function get_edge_stalk(s::EuclideanSheaf, v1::Int, v2::Int)
    edge_key = UnorderedPair(v1, v2)
    @assert haskey(s.edge_stalks, edge_key)
    return s.edge_stalks[edge_key]
end

"""    get_restriction_map(s::EuclideanSheaf, v1::Int, v2::Int)

Get the restriction map from vertex v1 to the edge (v1, v2).
"""
function get_restriction_map(s::EuclideanSheaf, v1::Int, v2::Int)
    @assert haskey(s.restriction_maps, v1 => v2)
    return s.restriction_maps[v1=>v2]
end

function add_vertex_stalk!(s::EuclideanSheaf, stalk_size::Int)
    push!(s.vertex_stalks, stalk_size)
    add_vertex!(s.underlying_graph)
end

"""   add_sheaf_edge!(s::EuclideanSheaf, v1::Int, v2::Int, rm1::Matrix{Float64}, rm2::Matrix{Float64})

Add an edge between vertices v1 and v2 with restriction maps rm1 and rm2.
rm1 is the restriction map from vertex v1 to the edge, and rm2 is the restriction map from vertex v2 to the edge.
"""
function add_sheaf_edge!(s::EuclideanSheaf{T}, v1::Int, v2::Int, rm1::Matrix{T}, rm2::Matrix{T}) where T
    @assert v1 <= length(s.vertex_stalks) && v2 <= length(s.vertex_stalks)
    @assert size(rm1)[1] == size(rm2)[1]
    @assert size(rm1)[2] == s.vertex_stalks[v1]
    @assert size(rm2)[2] == s.vertex_stalks[v2]

    stalk_size = size(rm1)[1]
    add_edge!(s.underlying_graph, v1, v2)
    edge_key = UnorderedPair(v1, v2)
    s.edge_stalks[edge_key] = stalk_size
    s.restriction_maps[v1=>v2] = rm1
    s.restriction_maps[v2=>v1] = rm2
    return ne(s.underlying_graph)
end

function coboundary_map(s::EuclideanSheaf{T}) where T
    I = Int64[]
    J = Int64[]
    V = Matrix{T}[]

    for (e_idx, e) in enumerate(edges(s.underlying_graph))
        i = src(e)
        j = dst(e)
        rm1 = s.restriction_maps[i=>j]
        rm2 = s.restriction_maps[j=>i]

        push!(J, i, j)
        push!(I, e_idx, e_idx)
        push!(V, rm1, -rm2)
    end
    return blocksparse(I, J, V)
end

function sheaf_laplacian(s::EuclideanSheaf)
    B = coboundary_map(s)
    return x -> B' * (B * x)
end

function sheaf_laplacian_matrix(s::EuclideanSheaf)
    B = coboundary_map(s)
    return B' * B
end


function energy_function(L::AbstractMatrix)
    return x -> 0.5 * x' * (L * x)
end

function energy_function(s::EuclideanSheaf)
    return energy_function(sheaf_laplacian_matrix(s))
end

function nearest_global_section(s::EuclideanSheaf, x; verbose=false)
    d = coboundary_map(s)

    eL = LinearOperator(d) * LinearOperator(d')

    b = d * x

    y, stats = cg(eL, Array(b))
    if verbose
        println(stats)
    end

    return BlockArray(x - d' * y, s.vertex_stalks)
end


end # EuclideanSheaves