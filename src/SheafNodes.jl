module SheafNodes

export SheafNode, add_neighbor!

using Distributed
using SparseArrays

mutable struct SheafNode
    id::Int32
    dimension::Int32
    neighbors::Dict{Int32, SparseMatrixCSC{Float32, Int32}}
    in_channels::Dict{Int32, RemoteChannel}
    out_channels::Dict{Int32, RemoteChannel}
    x::Vector{Float32}
end

function add_neighbor!(s::SheafNode, n_id::Int32, restriction_map)
    s.neighbors[n_id] = restriction_map
end

function neighbors(s::SheafNode)
    return collect(keys(s.neighbors))
end

end