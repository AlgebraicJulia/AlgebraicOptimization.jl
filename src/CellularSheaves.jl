module CellularSheaves

export CellularSheaf, add_map!, coboundary_map, laplacian, is_global_section, SheafObjective, apply_f, apply_f_with_stabilizer, apply_lagrangian_to_x, apply_lagrangian_to_z, simulate!

using BlockArrays
using ForwardDiff
using Distributed


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


function simulate!(so::SheafObjective, λ::Float64 = .1, n_steps::Int = 100)  # Uzawa's algorithm. Currently not very distributed.
    for _ in 1:n_steps
        x_update = -ForwardDiff.gradient(apply_lagrangian_to_x(so), so.x)   # Ideally, these should be done separately for the individual x's.
        z_update = ForwardDiff.gradient(apply_lagrangian_to_z(so), so.z)   # Ideally, these should be done separately for the individual z's.
        
        so.x += x_update * λ
        so.z += z_update * λ
        # println(z_update - laplacian(so.s) * so.x)   <-- Sanity check; always should be 0
    end
end


# mutable struct SheafNode
#     adj::Vector{Int}    # Indices of neighboring SheafNodes
#     maps::Vector{Array}   # maps[i] corresponds to the resrtiction map from the SheafNode to the edge connecting it to adj[i]
#     f::Function    # Objective function at vertex
#     x::Vector  # Primary variables
#     z::Vector  # Dual variables
# end


mutable struct SheafNode
    adj::Dict{SheafNode, Array}
    f::Function    # Objective function at vertex
    x::Vector  # Primary variables
    z::Vector  # Dual variables
end

function SheafNode(f::Function, x::Vector, z::Vector)
    return SheafNode(Dict(), f, x, z)
end

function add_edge!(u::SheafNode, v::SheafNode, u_map::Array, v_map::Array)
    u.adj[v] = u_map
    v.adj[u] = v_map
end


function simulate!(sheaf::Vector{SheafNode}, λ::Float64 = .1, n_steps::Int = 100)   # Could make a Sheaf wrapper for Vector{SheafNode} for cleanliness
    for _ in 1:n_steps
        for v::SheafNode in sheaf
            x_update = zeros(length(v.x))
            z_update = zeros(length(v.z))
            for u in keys(v.adj)
                # Get F
                x_update +=  -ForwardDiff.gradient(v.f, v.x) -2 * v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x) - v.adj[u]' * (v.adj[u] * v.z - u.adj[v] * u.z)
                z_update += v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x)
            end
            v.x += x_update * λ
            v.z += z_update * λ
        end
    end
end

function simulate_distributed!(sheaf::Vector{SheafNode}, λ::Float64 = .1, n_steps::Int = 100)   # Could eliminate the sheaf node part. Also, should we alternate x and z updates?
    for _ in 1:n_steps
        @sync @distributed for v::SheafNode in sheaf
            x_update = zeros(length(v.x))
            z_update = zeros(length(v.z))
            for u in keys(v.adj)
                # Get F
                x_update +=  -ForwardDiff.gradient(v.f, v.x) -2 * v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x) - v.adj[u]' * (v.adj[u] * v.z - u.adj[v] * u.z)
                z_update += v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x)
            end
            v.x += x_update * λ
            v.z += z_update * λ
        end
    end
end



end      # module