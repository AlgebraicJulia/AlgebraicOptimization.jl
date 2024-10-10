module CellularSheaves

export CellularSheaf, add_map!, laplacian, is_global_section

using LinearAlgebra  # Required for Matrix type
using SparseArrays
using BlockArrays

struct CellularSheaf
    V::Int    # Number of vertices
    E::Int    # Number of edges
    N::Int    # Max dimension of any stalk
    restriction_maps::BlockArray    # E*N x V*N matrix. Viewed as a block matrix with N*N blocks, the (e, v) entry represents the restriction map from v -> e.
end

# Constructor: no coboundary map given
function CellularSheaf(V::Int, E::Int, N::Int)
    restriction_maps = zeros(Float64, E * N, V * N)

    row_blocks = fill(N, E)  # E blocks of N rows each
    col_blocks = fill(N, V)  # V blocks of N columns each

    # Create a BlockArray from restriction_maps
    block_matrix = BlockArray(restriction_maps, row_blocks, col_blocks)
    CellularSheaf(V, E, N, block_matrix)
end

function add_map!(s::CellularSheaf, v::Int, e::Int, map::Matrix{Float64})
    s.restriction_maps[Block(e, v)] = map
end

function laplacian(s::CellularSheaf)
    return s.restriction_maps' * s.restriction_maps
end

function is_global_section(s::CellularSheaf, v::Vector)
    return iszero(laplacian(s) * v)      # This may only work if the graph underlying s is connected
end

end      # module