module CellularSheaves

export CellularSheaf, add_map!, coboundary_map, laplacian, is_global_section, SheafObjective, apply_f, apply_f_with_stabilizer, apply_lagrangian_to_x, apply_lagrangian_to_z, simulate!

using BlockArrays
using ForwardDiff


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


mutable struct SheafObjective
    objectives::Vector{Function}    # Number of vertices
    s::CellularSheaf    # E*N x V*N matrix. Viewed as a block matrix with N*N blocks, the (e, v) entry represents the restriction map from v -> e.
    x::Vector   # Primary variables
    z::Vector   # Dual variables
end


function apply_f(so::SheafObjective)
    return sum(so.objectives[i](so.x[i]) for i in eachindex(so.x))
end

function apply_f_with_stabilizer(so::SheafObjective)
    return sum(so.objectives[i](so.x[i]) for i in eachindex(so.x))   # + (so.x)' * laplacian(so.s) * so.x   <-- Does this term change the answer?
end

function lagrangian(so::SheafObjective)
    return sum(so.objectives[i](so.x[i]) for i in eachindex(so.x)) + (so.x)' * laplacian(so.s) * so.x + (so.z)' * laplacian(so.s) * so.x
end


# The following 2 functions are necessary for taking the differential of the L(x, z) with respect to x and z using ForwardDiff.
function apply_lagrangian_to_x(so::SheafObjective)
    return x -> sum(so.objectives[i](x[i]) for i in eachindex(x)) + (x)' * laplacian(so.s) * x + so.z' * laplacian(so.s) * x
end

function apply_lagrangian_to_z(so::SheafObjective)
    return z -> sum(so.objectives[i](so.x[i]) for i in eachindex(so.x)) + (so.x)' * laplacian(so.s) * so.x + z' * laplacian(so.s) * so.x
end


function simulate!(so::SheafObjective, λ::Float64 = .1, n_steps::Int = 100)  # Like Uzawa's algorithm. Currently not very distributed.
    for _ in 1:n_steps
        x_update = -ForwardDiff.gradient(apply_lagrangian_to_x(so), so.x)   # Ideally, these should be done separately for the individual x's.
        so.x += x_update * λ

        z_update = ForwardDiff.gradient(apply_lagrangian_to_z(so), so.z)   # Ideally, these should be done separately for the individual z's.
        so.z += z_update * λ
    end
end

end      # module