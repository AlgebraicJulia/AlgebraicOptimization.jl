module CellularSheaves

export CellularSheaf, add_map!, coboundary_map, laplacian, is_global_section, SheafObjective, apply_f, apply_f_with_stabilizer, apply_lagrangian_to_x, apply_lagrangian_to_z, simulate!,
    SheafNode, add_edge!, simulate_distributed!, simulate_distributed_separate_steps!, SheafVertex, SheafEdge, xLaplacian, zLaplacian, ThreadedSheaf, simulate_sequential!

using BlockArrays
using ForwardDiff
using Distributed


struct CellularSheaf
    V::Vector{Int}    # Vertex dimensions
    E::Vector{Int}    # Edge dimensions
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

function add_map!(s::CellularSheaf, v::Int, e::Int, map::Matrix{})
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
    objectives::Vector{Function}    # List of objective functions at each vertex
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


# First distriubted implementation: maps from neighboring SheafNodes to the restriction maps into the shared edges
mutable struct SheafNode
    adj::Dict{SheafNode, Array}  # v   ---- e  ------ w   is stored as   {w -> (v -> e)}
    f::Function    # Objective function at vertex
    x::Vector  # Primary variables
    z::Vector  # Dual variables
    x_update::Vector
    z_update::Vector
end

function SheafNode(f::Function, x::Vector, z::Vector)
    return SheafNode(Dict(), f, x, z, x, z)
end

function add_edge!(u::SheafNode, v::SheafNode, u_map::Array, v_map::Array)
    u.adj[v] = u_map
    v.adj[u] = v_map
end


# Uses the distributed sheaf structure to do Uzawa's algorithm, but executes sequentially (not distributed)
function simulate!(sheaf::Vector{SheafNode}, λ::Float64 = .1, n_steps::Int = 10) 
    for _ in 1:n_steps
        for i in 1:length(sheaf)
            v = sheaf[i]
            x_update = zeros(length(v.x))
            z_update = zeros(length(v.z))

            # Compute updates
            x_update +=  -ForwardDiff.gradient(v.f, v.x)
            for u in keys(v.adj)
                x_update += -2 * v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x) - v.adj[u]' * (v.adj[u] * v.z - u.adj[v] * u.z)
                z_update += v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x)
            end

            # Apply updates
            v.x += x_update * λ
            v.z += z_update * λ
        end
    end
end


# Same as above but uses @distributed to actually execute in a distributed way
function simulate_distributed!(sheaf::Vector{SheafNode}, λ::Float64 = .1, n_steps::Int = 10)   
    for _ in 1:n_steps
        @sync @distributed for i in 1:length(sheaf) 
            v = sheaf[i]
            x_update = zeros(length(v.x))
            z_update = zeros(length(v.z))

            # Compute updates
            x_update +=  -ForwardDiff.gradient(v.f, v.x)
            for u in keys(v.adj)
                x_update += -2 * v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x) - v.adj[u]' * (v.adj[u] * v.z - u.adj[v] * u.z)
                z_update += v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x)
            end

            # Apply updates
            v.x += x_update * λ
            v.z += z_update * λ
        end
    end
end


# Same Uzawa's algorithm, except computing and applying updates are separated into different steps. More suited to the literature.
function simulate_distributed_separate_steps!(sheaf::Vector{SheafNode}, λ::Float64 = .1, n_steps::Int = 10) 
    for _ in 1:n_steps
        # Phase 1: Compute updates
        @sync @distributed for i in 1:length(sheaf)   # Separate read/compute and write steps? Might be helpful...     
            v = sheaf[i]
            v.x_update .= 0  # Fill with zeros
            v.z_update .- 0  # Fill with zeros

            # Compute updates
            v.x_update +=  -ForwardDiff.gradient(v.f, v.x)
            for u in keys(v.adj)
                v.x_update += -2 * v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x) - v.adj[u]' * (v.adj[u] * v.z - u.adj[v] * u.z)
                v.z_update += v.adj[u]' * (v.adj[u] * v.x - u.adj[v] * u.x)
            end
        end

        # Phase 2: Apply updates
        @sync @distributed for v in sheaf 
                # Apply updates
                v.x += v.x_update * λ
                v.z += v.z_update * λ
        end
    end
end



# Dr. Fairbanks' approach: Separate vertex and edge workers

mutable struct SheafVertex
    adj::Vector{}  # It's actually a Vector{SheafEdge} but this was causing a circular dependency problem. Use Vector{Int} instead?
    f::Function    # Objective function at vertex
    x::Vector  # Primary variables
    z::Vector  # Dual variables
    x_update::Vector
    z_update::Vector
    # x_update::Vector   # This would be if we wanted to separate the compute and update steps, as above
    # z_update::Vector
end

struct SheafEdge    # Note-- not mutable!
    src::SheafVertex         # How to separate out src, target? Also, should src be an actual sheafnode, not just an int?
    tgt::SheafVertex
    left::Array
    right::Array
end

function SheafVertex(f::Function, x::Vector, z::Vector)
    return SheafVertex([], f, x, z, zeros(length(x)), zeros(length(z)))
end

function add_edge!(u::SheafVertex, v::SheafVertex, u_map::Array, v_map::Array)
    edge = SheafEdge(u, v, u_map, v_map)
    push!(u.adj, edge)
    push!(v.adj, edge)
end

# "Asks the shared edge" to calculate the local update to the primal x variable
function xLaplacian(v::SheafVertex, e::SheafEdge)
    if v == e.src
        return -2 * e.left' * (e.left * e.src.x - e.right * e.tgt.x) - e.left' * (e.left * e.src.z - e.right * e.tgt.z)
    else
        return 2 * e.right' * (e.left * e.src.x - e.right * e.tgt.x) + e.right' * (e.left * e.src.z - e.right * e.tgt.z)
    end
end

# "Asks the shared edge" to calculate the local update to the dual z variable
function zLaplacian(v::SheafVertex, e::SheafEdge)
    if v == e.src
        return e.left' * (e.left * e.src.x - e.right * e.tgt.x)
    else
        return -1 * e.right' * (e.left * e.src.x - e.right * e.tgt.x)
    end
end

# This newer implementation with the separate vertex and edge workers makes the simulate method cleaner
function simulate_distributed!(sheaf::Vector{SheafVertex}, λ::Float64 = .1, n_steps::Int = 10)  
    for _ in 1:n_steps
        @sync @distributed for i in 1:length(sheaf)  # Does this actually work, or are different processes getting different information?
            v = sheaf[i]
            v.x_update .= 0   # Fill with zeros to reset update
            v.z_update .= 0   # Fill with zeros to reset update

            # Compute updates
            v.x_update +=  -ForwardDiff.gradient(v.f, v.x)
            for e in v.adj
                v.x_update += xLaplacian(v, e)
                v.z_update += zLaplacian(v, e)
            end
            v.x += v.x_update * λ
            v.z += v.z_update * λ
        end
    end
end










# New approach: Threading & shared memory
# Should be very straightforward

mutable struct ThreadedSheaf
    x::BlockArray{Float64} 
    λ::BlockArray{Float64}
    f::Vector{Function}
    restriction_maps::BlockArray{Float64}  # Could be a sparse array. This is an e * v matrix.
end




# Constructor: no coboundary map given
function ThreadedSheaf(V::Vector{Int}, E::Vector{Int}, f::Union{Vector{Function}, Nothing} = nothing)
    f = f === nothing ? Function[] : f  # Set `f` to an empty vector if `nothing` was provided
    x = BlockArray{Float64}(ones(sum(V), 1), V, [1])
    λ = BlockArray{Float64}(zeros(sum(V), 1), V, [1])
    restriction_maps = BlockArray{Float64}(zeros(sum(E), sum(V)), E, V)
    return ThreadedSheaf(x, λ, f, restriction_maps)
end





# # Constructor: no coboundary map given, constant dimension
# function CellularSheaf(num_V::Int, num_E::Int, max_dimension::Int)
#     V = fill(max_dimension, num_V)  # E blocks of N rows each
#     E = fill(max_dimension, num_E)  # V blocks of N columns each
#     return CellularSheaf(V, E)
# end


function add_map!(s::ThreadedSheaf, v::Int, e::Int, map::Matrix{})
    s.restriction_maps[Block(e, v)] = map
end

function coboundary_map(s::ThreadedSheaf)
    # Iterate through the restriction_maps matrix and negate the second non-zero block in each row
    coboundary = copy(s.restriction_maps)
    for e in 1:blocksize(coboundary)[1]   # Iterate the block rows
        has_seen_block = false
        for v in 1:blocksize(coboundary)[2]
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


function laplacian(s::ThreadedSheaf)
    return coboundary_map(s)' * coboundary_map(s)
end

function is_global_section(s::ThreadedSheaf)   # This should now just be based off of the threaded sheaf, since it includes the info
    return iszero(laplacian(s) * s.x)      # This may only work if the graph underlying s is connected
end


function simulate!(s::ThreadedSheaf, α::Float64 = .1, n_steps::Int = 1000)  # Uzawa's algorithm. Currently not very distributed.
    L = laplacian(s)
    for _ in 1:n_steps
        # Gradient update step
        Threads.@threads for v in 1:blocksize(s.x)[1]   # Iterate the vertices. This will be @threads.
            s.x[Block(v, 1)] += -ForwardDiff.gradient(s.f[v], s.x[Block(v, 1)]) * α
        end

        # Laplacian multiply step
        s.x +=  α * (-2 * L * s.x - L * s.λ)
        s.λ += α * L * s.x
    end
end

function simulate_sequential!(s::ThreadedSheaf, α::Float64 = .1, n_steps::Int = 1000)  # Uzawa's algorithm. Currently not very distributed.
    L = laplacian(s)
    for _ in 1:n_steps
        # Gradient update step
        for v in 1:blocksize(s.x)[1]   # Iterate the vertices. No @threads here.
            s.x[Block(v, 1)] += -ForwardDiff.gradient(s.f[v], s.x[Block(v, 1)]) * α
        end

        # Laplacian multiply step
        s.x +=  α * (-2 * L * s.x - L * s.λ)
        s.λ += α * L * s.x
    end
end


end      # module