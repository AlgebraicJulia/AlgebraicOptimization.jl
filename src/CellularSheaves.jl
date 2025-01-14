module CellularSheaves

export CellularSheaf, add_map!, coboundary_map, laplacian, is_global_section, SheafObjective, apply_f, apply_f_with_stabilizer, apply_lagrangian_to_x, apply_lagrangian_to_z, simulate!,
    SheafNode, simulate_distributed!, simulate_distributed_separate_steps!, SheafVertex, SheafEdge, xLaplacian, zLaplacian, MatrixSheaf, optimize!,
    OptimizationAlgorithm

import Catlab: add_edge!

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

# TODO: make a better API for adding edges based on pairs of vertices and maps.

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


function simulate!(so::SheafObjective, λ::Float64=0.1, n_steps::Int=100)  # Uzawa's algorithm. Currently not very distributed.
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
    adj::Dict{SheafNode,Array}  # v   ---- e  ------ w   is stored as   {w -> (v -> e)}
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
function simulate!(sheaf::Vector{SheafNode}, λ::Float64=0.1, n_steps::Int=10)
    for _ in 1:n_steps
        for i in 1:length(sheaf)
            v = sheaf[i]
            x_update = zeros(length(v.x))
            z_update = zeros(length(v.z))

            # Compute updates
            x_update += -ForwardDiff.gradient(v.f, v.x)
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
function simulate_distributed!(sheaf::Vector{SheafNode}, λ::Float64=0.1, n_steps::Int=10)
    for _ in 1:n_steps
        @sync @distributed for i in 1:length(sheaf)
            v = sheaf[i]
            x_update = zeros(length(v.x))
            z_update = zeros(length(v.z))

            # Compute updates
            x_update += -ForwardDiff.gradient(v.f, v.x)
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
function simulate_distributed_separate_steps!(sheaf::Vector{SheafNode}, λ::Float64=0.1, n_steps::Int=10)
    for _ in 1:n_steps
        # Phase 1: Compute updates
        @sync @distributed for i in 1:length(sheaf)   # Separate read/compute and write steps? Might be helpful...     
            v = sheaf[i]
            v.x_update .= 0  # Fill with zeros
            v.z_update .- 0  # Fill with zeros

            # Compute updates
            v.x_update += -ForwardDiff.gradient(v.f, v.x)
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
function simulate_distributed!(sheaf::Vector{SheafVertex}, λ::Float64=0.1, n_steps::Int=10)
    for _ in 1:n_steps
        @sync @distributed for i in 1:length(sheaf)  # Does this actually work, or are different processes getting different information?
            v = sheaf[i]
            v.x_update .= 0   # Fill with zeros to reset update
            v.z_update .= 0   # Fill with zeros to reset update

            # Compute updates
            v.x_update += -ForwardDiff.gradient(v.f, v.x)
            for e in v.adj
                v.x_update += xLaplacian(v, e)
                v.z_update += zLaplacian(v, e)
            end
            v.x += v.x_update * λ
            v.z += v.z_update * λ
        end
    end
end



"""    
    mutable struct MatrixSheaf

A data structure representing a cellular sheaf, which includes primal and dual variables, objective functions, and restriction maps. 
This is primarily used for distributed Uzawa-type optimization algorithms. Based on shared memory and multithreading.

# Fields:
- `x::BlockArray{Float64}`: Primal variables.
- `λ::BlockArray{Float64}`: Dual variables (Lagrange multipliers).
- `f::Vector{Function}`: Objective functions associated with vertices.
- `restriction_maps::BlockArray{Float64}`: Restriction maps, stored as an `e × v` matrix.
- `coboundary::BlockArray{Float64}`: Coboundary maps derived from the restriction maps.
"""
mutable struct MatrixSheaf
    x::BlockArray{Float64} 
    λ::BlockArray{Float64}
    f::Vector{Function}
    restriction_maps::BlockArray{Float64}  # Could be a sparse array. This is an e * v matrix.
    coboundary::BlockArray{Float64}
end




"""    
    MatrixSheaf(V::Vector{Int}, E::Vector{Int}, f::Union{Vector{Function}, Nothing} = nothing)

Constructs a `MatrixSheaf` with specified vertex sizes `V` and edge sizes `E`. If objective functions `f` are not provided, an empty vector is used.

# Arguments:
- `V::Vector{Int}`: Dimensions of stalks corresponding to vertices.
- `E::Vector{Int}`: Dimensions of stalks corresponding to edges.
- `f::Union{Vector{Function}, Nothing}`: Optional vector of objective functions.
"""
function MatrixSheaf(V::Vector{Int}, E::Vector{Int}, f::Union{Vector{Function}, Nothing} = nothing)
    f = f === nothing ? Function[] : f  # Set `f` to an empty vector if `nothing` was provided
    x = BlockArray{Float64}(ones(sum(V), 1), V, [1])
    λ = BlockArray{Float64}(zeros(sum(V), 1), V, [1])
    restriction_maps = BlockArray{Float64}(zeros(sum(E), sum(V)), E, V)
    return MatrixSheaf(x, λ, f, restriction_maps, restriction_maps)
end



"""
    add_map!(s::MatrixSheaf, v::Int, e::Int, map::Matrix)

Adds a restriction map to a threaded sheaf from vertex e to edge e.

# Arguments:
- `s::MatrixSheaf`: The sheaf to which the map is added.
- `v::Int`: Index of the vertex block.
- `e::Int`: Index of the edge block.
- `map::Matrix`: The restriction map to insert.
"""
function add_map!(s::MatrixSheaf, v::Int, e::Int, map::Matrix{})
    s.restriction_maps[Block(e, v)] = map
end



"""
    coboundary_map(s::MatrixSheaf) -> BlockArray{Float64}

Computes the coboundary map for a threaded sheaf. Negates the second non-zero block in each row.
"""
function coboundary_map(s::MatrixSheaf)
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
# TODO: Just store the coboundary_map. Also, that should definitely be sparse! Nice.


"""
    laplacian(s::MatrixSheaf) -> BlockArray{Float64}

Computes the sheaf Laplacian matrix as the product of the transpose of the coboundary map and the coboundary map itself.
"""
function laplacian(s::MatrixSheaf)
    return coboundary_map(s)' * coboundary_map(s)
end


"""
    make_coboundary(s::MatrixSheaf)

Updates the `coboundary` field of the threaded sheaf with the computed coboundary map based on the current restriction maps.
"""
function make_coboundary(s::MatrixSheaf)
    s.coboundary = coboundary_map(s)
end



"""
    is_global_section(s::MatrixSheaf; tol::Float64 = 1e-8) -> Bool

Checks if the sheaf is a global section by verifying if the product of the Laplacian and primal variables is approximately zero.
"""
function is_global_section(s::MatrixSheaf; tol::Float64 = 1e-8)
    # Check if the product of the Laplacian and s.x is approximately zero within the given tolerance
    return isapprox(s.coboundary' * s.coboundary * s.x, zero(s.x); atol=tol)
end


"""
    random_threaded_sheaf(V::Int, E::Int, dim::Int) -> MatrixSheaf

Generates a random threaded sheaf with vertices, edges, and dimension.

# Arguments:
- `V::Int`: Number of vertices.
- `E::Int`: Number of edges.
- `dim::Int`: Dimension of the stalks on every vertex and edge.

# Returns:
- `MatrixSheaf`: A randomly initialized threaded sheaf.
"""
function random_threaded_sheaf(V::Int, E::Int, dim::Int)
    random_sheaf = MatrixSheaf([dim for _ in 1:V], [dim for _ in 1:E])

    # Add random restriction maps
    for e in 1:E
        u = rand(1:V)
        w = rand(1:V)
        while w == u
            w = rand(1:V)
        end
        add_map!(random_sheaf, u, e, rand(dim, dim))
        add_map!(random_sheaf, w, e, rand(dim, dim))
    end

    # Add random objective functions
    random_sheaf.f = [x -> only(x' * Q * x + b * x) for _ in 1:V for Q = [rand(dim, dim)], b = [rand(1, dim)]]   # TODO: Q is positive semidefinite. Q = Q^T Q
    # TODO: Vary the dimensions
    # TODO: Use Catlab's random graph function. Take a graph and add random restriction maps
    return random_sheaf
end


function simulate!(s::MatrixSheaf, α::Float64 = .1, n_steps::Int = 1000)  # Uzawa's algorithm. Currently not very distributed.
    make_coboundary(s)
    for _ in 1:n_steps
        # Gradient update step
        Threads.@threads for v in 1:blocksize(s.x)[1]   # Iterate the vertices. This will be @threads.
            s.x[Block(v, 1)] += -ForwardDiff.gradient(s.f[v], s.x[Block(v, 1)]) * α  
            # TODO: Turn ForwardDiff into ReverseDiff
        end

        # Laplacian multiply step
        s.x +=  α * (-2 * s.coboundary' * s.coboundary * s.x - s.coboundary' * s.coboundary * s.λ)
        s.λ += α * s.coboundary' * s.coboundary * s.x
    end
end

function simulate_sequential!(s::MatrixSheaf, α::Float64 = .1, n_steps::Int = 1000)  # Uzawa's algorithm. Currently not very distributed.
    s.coboundary = coboundary_map(s)   # Calculate the coboundary map based on restriction maps
    for _ in 1:n_steps
        # Gradient update step
        for v in 1:blocksize(s.x)[1]   # Iterate the vertices. No @threads here.
            s.x[Block(v, 1)] += -ForwardDiff.gradient(s.f[v], s.x[Block(v, 1)]) * α   # TODO: Research faster gradient methods
        end

        # Laplacian multiply step
        s.x +=  α * s.coboundary' * s.coboundary * (-2 * s.x - s.λ)
        s.λ += α * s.coboundary' * s.coboundary * s.x
    end
end

# Diagonal dominant
# Add stuff to the diagonal (+xI)   smallest abs. value of the row sum of a symmetric matrix -> add that to the diagonal





# Unified interface to optimize sheaf objectives using different algorithmic backends

abstract type OptimizationAlgorithm end

struct Uzawas <: OptimizationAlgorithm
    step_size::Float64
    max_iters::Float64
    epsilon::Float64 # Terminate if objective value decreases by less than epsilon in a given iteration
end

function optimize!(s::SheafObjective, alg::Uzawas)
    simulate!(s, alg.step_size, alg.max_iters)
    # TODO: implement convergence test.
end


end      # module