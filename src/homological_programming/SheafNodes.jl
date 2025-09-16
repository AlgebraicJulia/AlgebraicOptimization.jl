module SheafNodes

export AbstractSheafNode, DistributedSheafNode, ThreadedSheafNode, AsyncSheafNode, add_neighbor!, neighbors

using Distributed
using SparseArrays

abstract type AbstractSheafNode end

#TODO: Refactor to make these not have explicit state
mutable struct ThreadedSheafNode <: AbstractSheafNode
    id::Int64
    dimension::Int64
    neighbors::Dict{Int64,AbstractMatrix}
    in_channels::Dict{Int64,Channel}
    out_channels::Dict{Int64,Channel}
    x::Vector{Float64}
end

mutable struct AsyncSheafNode <: AbstractSheafNode
    id::Int64
    dimension::Int64
    neighbors::Dict{Int64,AbstractMatrix}
    in_channels::Dict{Int64,Channel}
    out_channels::Dict{Int64,Channel}
    x::Vector{Float64}
    period::Int64
    phase::Int64
    iteration::Int64
    # itersSinceUpdate::Int64
end

mutable struct DistributedSheafNode <: AbstractSheafNode
    id::Int64
    dimension::Int64
    neighbors::Dict{Int64,AbstractMatrix}
    in_channels::Dict{Int64,RemoteChannel}
    out_channels::Dict{Int64,RemoteChannel}
    x::Vector{Float64}
end

function add_neighbor!(s::AbstractSheafNode, n_id::Int64, restriction_map)
    s.neighbors[n_id] = restriction_map
end

function neighbors(s::AbstractSheafNode)
    return collect(keys(s.neighbors))
end

end