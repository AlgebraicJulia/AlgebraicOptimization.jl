module SheafInterface

export AbstractNetworkSheaf, vertex_stalks, edge_stalks, coboundary_map, add_vertex_stalk!, add_sheaf_edge!, underlying_graph,
    get_vertex_stalk, get_edge_stalk, get_restriction_map, sheaf_laplacian

import Base: show

"""     AbstractNetworkSheaf

An abstract type for network sheaves.
A network sheaf is a cellular sheaf on a graph, i.e. a sheaf on a 1-dimensional cell complex.
    It consists of:
    - A graph G = (V, E)
    - A stalk S_v for each vertex v in V
    - A stalk S_e for each edge e in E
    - A restriction map r_{e->v} : S_e -> S_v for each incident vertex-edge pair (v, e)
"""
abstract type AbstractNetworkSheaf end

function vertex_stalks(s::AbstractNetworkSheaf)
    error("vertex_stalks not implemented")
end

function edge_stalks(s::AbstractNetworkSheaf)
    error("edge_stalks not implemented")
end

function underlying_graph(s::AbstractNetworkSheaf)
    error("underlying_graph not implemented")
end

function get_vertex_stalk(s::AbstractNetworkSheaf, v)
    error("get_vertex_stalk not implemented")
end

function get_edge_stalk(s::AbstractNetworkSheaf, v1, v2)
    error("get_edge_stalk not implemented")
end

function get_restriction_map(s::AbstractNetworkSheaf, v1, v2)
    error("get_restriction_map not implemented")
end


function add_vertex_stalk!(s::AbstractNetworkSheaf, stalk)
    error("add_vertex_stalk! not implemented")
end

function add_sheaf_edge!(s::AbstractNetworkSheaf, v1, v2, rm1, rm2)
    error("add_sheaf_edge! not implemented")
end

function coboundary_map(s::AbstractNetworkSheaf)
    error("coboundary_map not implemented")
end

function sheaf_laplacian(s::AbstractNetworkSheaf)
    error("laplacian not implemented")
end

function Base.show(io::IO, s::AbstractNetworkSheaf)
    println(io, "A network sheaf with ", length(vertex_stalks(s)), " vertex stalks and ", length(edge_stalks(s)), " edge stalks.")
end

end