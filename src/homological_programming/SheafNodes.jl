module SheafNodes

export DistributedSheafNode, ThreadedSheafNode, add_neighbor!, neighbors

using Distributed
using SparseArrays

abstract type AbstractSheafNode end

#TODO: Refactor to make these not have explicit state
mutable struct ThreadedSheafNode <: AbstractSheafNode
    id::Int32
    dimension::Int32
    neighbors::Dict{Int32,AbstractMatrix}
    in_channels::Dict{Int32,Channel}
    out_channels::Dict{Int32,Channel}
    x::Vector{Float32}
end

mutable struct DistributedSheafNode <: AbstractSheafNode
    id::Int32
    dimension::Int32
    neighbors::Dict{Int32,AbstractMatrix}
    in_channels::Dict{Int32,RemoteChannel}
    out_channels::Dict{Int32,RemoteChannel}
    x::Vector{Float32}
end

function add_neighbor!(s::AbstractSheafNode, n_id::Int32, restriction_map)
    s.neighbors[n_id] = restriction_map
end

function neighbors(s::AbstractSheafNode)
    return collect(keys(s.neighbors))
end

end