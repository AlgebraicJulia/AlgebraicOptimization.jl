module PotentialSheaves

export PotentialSheaf, get_edge_potential

using ..EuclideanSheaves
using ..SheafInterface
import ..SheafInterface: vertex_stalks, edge_stalks, coboundary_map, add_vertex_stalk!, add_sheaf_edge!, underlying_graph,
    get_vertex_stalk, get_edge_stalk, get_restriction_map, sheaf_laplacian
using ForwardDiff
using BlockArrays
using AutoHashEquals
using Graphs

@auto_hash_equals struct PotentialSheaf{S<:AbstractNetworkSheaf} <: AbstractNetworkSheaf
    sheaf::S
    edge_potentials::Dict{UnorderedPair{Int},Function}
end

PotentialSheaf(s::EuclideanSheaf) = PotentialSheaf(s, Dict{UnorderedPair{Int},Function}())

PotentialSheaf{T}(vertex_stalks::Vector{Int}) where T = PotentialSheaf(T(vertex_stalks), Dict{UnorderedPair{Int},Function}())

function vertex_stalks(s::PotentialSheaf)
    return vertex_stalks(s.sheaf)
end

function edge_stalks(s::PotentialSheaf)
    return edge_stalks(s.sheaf)
end

function underlying_graph(s::PotentialSheaf)
    return underlying_graph(s.sheaf)
end

function get_vertex_stalk(s::PotentialSheaf, v::Int)
    return get_vertex_stalk(s.sheaf, v)
end

function get_edge_stalk(s::PotentialSheaf, v1::Int, v2::Int)
    return get_edge_stalk(s.sheaf, v1, v2)
end

function get_restriction_map(s::PotentialSheaf, v1::Int, v2::Int)
    return get_restriction_map(s.sheaf, v1, v2)
end

function get_edge_potential(s::PotentialSheaf, v1::Int, v2::Int)
    k = UnorderedPair(v1, v2)
    if haskey(s.edge_potentials, k)
        return s.edge_potentials[k]
    else
        return x -> 0.5 * x' * x # Default potential is quadratic
    end
end

function add_vertex_stalk!(s::PotentialSheaf, stalk)
    add_vertex_stalk!(s.sheaf, stalk)
end

function add_sheaf_edge!(s::PotentialSheaf, v1::Int, v2::Int, rm1::Matrix{Float64}, rm2::Matrix{Float64}, potential::Function)
    add_sheaf_edge!(s.sheaf, v1, v2, rm1, rm2)
    s.edge_potentials[UnorderedPair(v1, v2)] = potential
end

function coboundary_map(s::PotentialSheaf)
    return coboundary_map(s.sheaf)
end

function sheaf_laplacian(s::PotentialSheaf; diff_backend=ForwardDiff)
    ordered_potentials = [s.edge_potentials[UnorderedPair(src(e), dst(e))] for e in edges(underlying_graph(s))]
    global_potential(y) = sum([potential(y[Block(e)]) for (e, potential) in enumerate(ordered_potentials)])
    B = coboundary_map(s)
    es = [edge_stalks(s)[UnorderedPair(src(e), dst(e))] for e in edges(underlying_graph(s))]
    L(x) = begin
        y = BlockArray(B * x, es)
        return B' * diff_backend.gradient(global_potential, y)
    end
    return L
end

end # module PotentialSheaves