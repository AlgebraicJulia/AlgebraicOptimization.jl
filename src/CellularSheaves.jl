module CellularSheaves

export CellularSheaf, add_map!, laplacian, is_global_section

using LinearAlgebra  # Required for Matrix type
using SparseArrays

struct CellularSheaf
    V::Int    # Number of vertices
    E::Int    # Number of edges
    N::Int    # Max dimension of any stalk
    coboundary_map::Matrix    # E*N x V*N matrix. Viewed as a block matrix with N*N blocks, the (e, v) entry represents the restriction map from v -> e.
end

# Constructor: no coboundary map given
function CellularSheaf(V::Int, E::Int, N::Int)
    CellularSheaf(V, E, N, zeros(Float64, E * N, V * N))
end

function add_map!(s::CellularSheaf, v::Int, e::Int, map::Matrix{Float64})
    println("here")
    dim = s.N
    for i in 1:(dim)
        for j in 1:(dim)
            println((e - 1) * dim + i)
            s.coboundary_map[(e - 1) * dim + i, (v - 1) * dim + j] = map[i, j];
        end
    end
end

function laplacian(s::CellularSheaf)
    return s.coboundary_map' * s.coboundary_map
end

function is_global_section(s::CellularSheaf, v::Vector)
    return iszero(laplacian(s) * v)      # This may only work if the graph underlying s is connected
end

end      # module