module CellularSheaves

export CellularSheaf, add_map!, coboundary_map, laplacian, is_global_section

using BlockArrays

struct CellularSheaf
    V::Vector{Int}    # Number of vertices
    E::Vector{Int}    # Number of edges
    # N::Vector{Int}    # Max dimension of any stalk
    restriction_maps::BlockArray    # E*N x V*N matrix. Viewed as a block matrix with N*N blocks, the (e, v) entry represents the restriction map from v -> e.
end


# Constructor: no coboundary map given
function CellularSheaf(V::Vector{Int}, E::Vector{Int})
    restriction_maps = BlockArray{Float64}(zeros(sum(E), sum(V)), E, V)
    return CellularSheaf(V, E, restriction_maps)
end

# Constructor: no coboundary map given, constant dimension
function CellularSheaf(num_V::Int, num_E::Int, max_dimension::Int)
    V = fill(max_dimension, num_V)  # E blocks of N rows each
    E = fill(max_dimension, num_E)  # V blocks of N columns each
    return CellularSheaf(V, E)
end

function add_map!(s::CellularSheaf, v::Int, e::Int, map::Matrix{Float64})
    s.restriction_maps[Block(e, v)] = map
end

function coboundary_map(s::CellularSheaf)
    # Iterate through the restriction_maps matrix and negate the second non-zero block in each row
    coboundary = copy(s.restriction_maps)
    for e in eachindex(s.E)
        has_seen_block = false
        for v in eachindex(s.V)
            if !has_seen_block && !(iszero(coboundary[Block(e, v)]))
                has_seen_block = true
            elseif has_seen_block && !(iszero(coboundary[Block(e, v)]))
                coboundary[Block(e, v)] = -1 * coboundary[Block(e, v)]
                break
            end
        end
    end
    return coboundary
end


function laplacian(s::CellularSheaf)
    return coboundary_map(s)' * coboundary_map(s)
end

function is_global_section(s::CellularSheaf, v::Vector)
    return iszero(laplacian(s) * v)      # This may only work if the graph underlying s is connected
end

end      # module