module ThreadedSheaves

export random_threaded_sheaf, random_async_threaded_sheaf, random_initialization, initialize!, compute_clusters, threaded_sheaf, laplacian_step!, iterate_laplacian!, iterate_laplacian_async!, distance_from_consensus, lipschitz_constant, coboundary_map

import ..DistributedSheaves: iterate_laplacian!, distance_from_consensus

using ..SheafNodes
using Base.Threads
using SparseArrays
using LinearAlgebra
using Graphs
using Metis
using ..CellularSheaves
using BlockArrays


function local_laplacian_step!(node::ThreadedSheafNode, step_size)
    x_old = node.x
    delta_x = zeros(node.dimension)

    for (n, rm) in node.neighbors
        outgoing_edge_val = rm * x_old
        incoming_edge_val = take!(node.in_channels[n])
        delta_x -= rm' * (outgoing_edge_val - incoming_edge_val)
    end
    x_new = x_old + step_size * delta_x

    for (n, rm) in node.neighbors
        put!(node.out_channels[n], rm * x_new)
    end

    node.x = x_new
end




function laplacian_step!(nodes::Vector{<:AbstractSheafNode}, step_size::Float64)
    Threads.@threads for node in nodes
        local_laplacian_step!(node, step_size)
    end
end

function laplacian_step!(nodes::Vector{<:AbstractSheafNode}, step_size::Float64, clusters::Vector{Vector{Int}})
    Threads.@threads for c in clusters
        for i in c
            local_laplacian_step!(nodes[i], step_size)
        end
    end
end


# Compute the local update direction for a given node
function local_descent_direction(node::ThreadedSheafNode)
    x_old = node.x
    delta_x = zeros(node.dimension)

    for (n, rm) in node.neighbors
        outgoing_edge_val = rm * x_old
        incoming_edge_val = fetch(node.in_channels[n])
        delta_x -= rm' * (outgoing_edge_val - incoming_edge_val)
    end
    return delta_x
end

function descent_direction!(nodes::Vector{ThreadedSheafNode}, results::Vector{Vector{Float64}})
    Threads.@threads for i in eachindex(nodes)
        results[i] = local_descent_direction(nodes[i])
    end
end

function descent_direction!(nodes::Vector{ThreadedSheafNode}, results::Vector{Vector{Float64}}, clusters::Vector{Vector{Int}})
    Threads.@threads for c in clusters
        for i in c
            results[i] = local_descent_direction(nodes[i])
        end
    end
end

function local_consensus_objective(node::ThreadedSheafNode, x::Vector{Float64})
    loss = 0.0
    for (n, rm) in node.neighbors
        outgoing_edge_val = rm * x
        incoming_edge_val = fetch(node.in_channels[n])
        loss += LinearAlgebra.norm_sqr(outgoing_edge_val - incoming_edge_val)
    end
    return loss
end

function consensus_objective(nodes::Vector{ThreadedSheafNode}, xs::Vector{Vector{Float64}}, clusters::Vector{Vector{Int}})
    losses = Vector{Float64}(undef, length(nodes))
    Threads.@threads for c in clusters
        for i in c
            losses[i] = local_consensus_objective(nodes[i], xs[i])
        end
    end
    return sum(losses)
end

function consensus_objective(nodes::Vector{ThreadedSheafNode}, xs::Vector{Vector{Float64}})
    losses = Vector{Float64}(undef, length(nodes))
    Threads.@threads for i in eachindex(nodes)
        losses[i] = local_consensus_objective(nodes[i], xs[i])
    end
    return sum(losses)
end

# Assumes length(x) == length(y) and all inner dimensions also match
function threaded_sum(x, y)
    n = length(x)
    m = length(x[1])
    res = repeat([Vector{Float64}(undef, m)], n)
    Threads.@threads for i in eachindex(x)
        res[i] = x[i] + y[i]
    end
    return res
end

function threaded_sum(x, y, clusters::Vector{Vector{Int}})
    n = length(x)
    m = length(x[1])
    res = repeat([Vector{Float64}(undef, m)], n)
    Threads.@threads for c in clusters
        for i in c
            res[i] = x[i] + y[i]
        end
    end
    return res
end

function line_search(nodes, delta_x::Vector{Vector{Float64}})
    @assert length(nodes) == length(delta_x)
    c = 0.5
    τ = 0.1
    ms = Vector{Float64}(undef, length(nodes))
    Threads.@threads for i in eachindex(delta_x)
        ms[i] = (-delta_x[i])' * delta_x[i]
    end
    m = sum(ms)
    #println(m)
    t = -c * m
    a = Float64(0.001)
    x = [node.x for node in nodes]
    while consensus_objective(nodes, x) - consensus_objective(nodes, threaded_sum(x, a .* delta_x)) < a * t
        a = τ * a
    end
    #println("Step size: $a")
    return a
end

function line_search(nodes, delta_x::Vector{Vector{Float64}}, clusters::Vector{Vector{Int}})
    @assert length(nodes) == length(delta_x)
    c = 0.5
    τ = 0.1
    ms = Vector{Float64}(undef, length(nodes))
    Threads.@threads for c in clusters
        for i in c
            ms[i] = (-delta_x[i])' * delta_x[i]
        end
    end
    m = sum(ms)
    #println(m)
    t = -c * m
    a = Float64(0.001)
    x = [node.x for node in nodes]
    while consensus_objective(nodes, x, clusters) - consensus_objective(nodes, threaded_sum(x, a .* delta_x, clusters), clusters) < a * t
        a = τ * a
    end
    #println("Step size: $a")
    return a
end

# Each node takes a step in the direction of a*delta_x.
# Also updates communication channels.
function update_nodes!(nodes, a, delta_x)
    Threads.@threads for i in eachindex(nodes)
        nodes[i].x += a * delta_x[i]
        for (n, rm) in nodes[i].neighbors
            # Update the buffers
            take!(nodes[i].in_channels[n])
            put!(nodes[i].out_channels[n], rm * nodes[i].x)
        end
    end
end

function update_nodes!(nodes, a, delta_x, clusters::Vector{Vector{Int}})
    Threads.@threads for c in clusters
        for i in c
            nodes[i].x += a * delta_x[i]
            for (n, rm) in nodes[i].neighbors
                # Update the buffers
                take!(nodes[i].in_channels[n])
                put!(nodes[i].out_channels[n], rm * nodes[i].x)
            end
        end
    end
end

# Overwrites delta_x
function laplacian_step!(nodes, delta_x::Vector{Vector{Float64}})
    descent_direction!(nodes, delta_x)
    step_size = line_search(nodes, delta_x)
    update_nodes!(nodes, step_size, delta_x)
end

function laplacian_step!(nodes, delta_x::Vector{Vector{Float64}}, clusters::Vector{Vector{Int}})
    descent_direction!(nodes, delta_x, clusters)
    step_size = line_search(nodes, delta_x, clusters)
    update_nodes!(nodes, step_size, delta_x, clusters)
end

function random_threaded_sheaf(num_nodes, edge_probability, restriction_map_dimension, restriction_map_density)
    nodes = ThreadedSheafNode[]
    coin()::Bool = rand() < edge_probability
    n, p = restriction_map_dimension, restriction_map_density
    for i in 1:num_nodes
        push!(nodes, ThreadedSheafNode(i, n,
            Dict{Int64,SparseMatrixCSC{Float64,Int64}}(),
            Dict{Int64,Channel}(),
            Dict{Int64,Channel}(), rand(n)))
    end

    for i in 1:num_nodes
        for j in i+1:num_nodes
            if coin()
                A = sprand(n, n, p)
                B = sprand(n, n, p)

                nodes[i].neighbors[j] = A
                nodes[j].neighbors[i] = B

                i_to_j_channel = Channel{Vector{Float64}}(2)
                j_to_i_channel = Channel{Vector{Float64}}(2)

                nodes[i].in_channels[j] = j_to_i_channel
                nodes[i].out_channels[j] = i_to_j_channel
                put!(i_to_j_channel, A * nodes[i].x)

                nodes[j].in_channels[i] = i_to_j_channel
                nodes[j].out_channels[i] = j_to_i_channel
                put!(j_to_i_channel, B * nodes[j].x)
            end
        end
    end
    return nodes
end

function random_threaded_sheaf(g::Graph, restriction_map_dimension, restriction_map_density)
    nodes = ThreadedSheafNode[]
    n, p = restriction_map_dimension, restriction_map_density

    for i in 1:nv(g)
        push!(nodes, ThreadedSheafNode(i, n,
            Dict{Int64,SparseMatrixCSC{Float64,Int64}}(),
            Dict{Int64,Channel}(),
            Dict{Int64,Channel}(), rand(n)))
    end

    for e in edges(g)
        A = sprand(n, n, p)
        B = sprand(n, n, p)

        i, j = src(e), dst(e)

        nodes[i].neighbors[j] = A
        nodes[j].neighbors[i] = B

        i_to_j_channel = Channel{Vector{Float64}}(2)
        j_to_i_channel = Channel{Vector{Float64}}(2)

        nodes[i].in_channels[j] = j_to_i_channel
        nodes[i].out_channels[j] = i_to_j_channel
        put!(i_to_j_channel, A * nodes[i].x)

        nodes[j].in_channels[i] = i_to_j_channel
        nodes[j].out_channels[i] = j_to_i_channel
        put!(j_to_i_channel, B * nodes[j].x)
    end

    return nodes
end

function compute_clusters(g::Graph, num_clusters::Int)
    p = Metis.partition(g, num_clusters)

    clusters = Vector{Vector{Int}}(undef, num_clusters)

    for i in 1:num_clusters
        clusters[i] = findall(x -> x == i, p)
    end

    return clusters
end

# function distance_from_consensus(nodes)
#     #total_distance = 0.0
#     node_distances = zeros(length(nodes))
#     #Threads.@threads for (i,node) in zip(collect(1:length(nodes)),nodes)
#     Threads.@threads for i in eachindex(nodes)
#         node_distance = 0.0
#         # There is some double counting happening in here but idrc
#         for ((_, in_channel), (_, out_channel)) in zip(nodes[i].in_channels, nodes[i].out_channels)
#             node_distance += norm(fetch(in_channel) - fetch(out_channel))
#         end
#         #total_distance += node_distance
#         node_distances[i] = node_distance
#     end
#     return sum(node_distances)
# end

function distance_from_consensus(nodes)
    total_distance = 0.0
    for i in eachindex(nodes)
        xi = nodes[i].x
        for (j, rm) in nodes[i].neighbors
            xj = nodes[j].x
            # Compare restriction maps applied to each node's state
            total_distance += norm(rm * xi - nodes[j].neighbors[i] * xj)
        end
    end
    return total_distance / 2  # divide by 2 to avoid double counting
end




function distance_from_consensus(nodes, clusters::Vector{Vector{Int}})
    #total_distance = 0.0
    node_distances = zeros(length(nodes))
    #Threads.@threads for (i,node) in zip(collect(1:length(nodes)),nodes)
    Threads.@threads for c in clusters
        for i in c
            node_distance = 0.0
            # There is some double counting happening in here but idrc
            for ((_, in_channel), (_, out_channel)) in zip(nodes[i].in_channels, nodes[i].out_channels)
                node_distance += norm(fetch(in_channel) - fetch(out_channel))
            end
            #total_distance += node_distance
            node_distances[i] = node_distance
        end
    end
    return sum(node_distances)
end

# Returns a list of distances from consensus over the iterations
function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, step_size, num_iters::Int)
    distances = Float64[]

    for _ in 1:num_iters
        push!(distances, distance_from_consensus(nodes))
        laplacian_step!(nodes, step_size)
    end
    push!(distances, distance_from_consensus(nodes))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, step_size, num_iters::Int, clusters::Vector{Vector{Int}})
    distances = Float64[]

    for _ in 1:num_iters
        push!(distances, distance_from_consensus(nodes, clusters))
        laplacian_step!(nodes, step_size, clusters)
    end
    push!(distances, distance_from_consensus(nodes, clusters))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, step_size, epsilon::Float64)
    distances = Float64[]

    while true
        push!(distances, distance_from_consensus(nodes))
        laplacian_step!(nodes, step_size)

        if distances[end] <= epsilon
            break
        end
    end
    push!(distances, distance_from_consensus(nodes))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, step_size, epsilon::Float64, clusters::Vector{Vector{Int}})
    distances = Float64[]

    while true
        push!(distances, distance_from_consensus(nodes, clusters))
        laplacian_step!(nodes, step_size, clusters)

        if distances[end] <= epsilon
            break
        end
    end
    push!(distances, distance_from_consensus(nodes, clusters))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, num_iters::Int)
    distances = Float64[]
    dimensions = (n -> n.dimension).(nodes)
    delta_x = [Vector{Float64}(undef, d) for d in dimensions]
    for _ in 1:num_iters
        push!(distances, distance_from_consensus(nodes))
        laplacian_step!(nodes, delta_x)


    end
    push!(distances, distance_from_consensus(nodes))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, num_iters::Int, clusters::Vector{Vector{Int}})
    distances = Float64[]
    dimensions = (n -> n.dimension).(nodes)
    delta_x = [Vector{Float64}(undef, d) for d in dimensions]
    for _ in 1:num_iters
        push!(distances, distance_from_consensus(nodes, clusters))
        laplacian_step!(nodes, delta_x, clusters)
    end
    push!(distances, distance_from_consensus(nodes, clusters))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, epsilon::Float64)
    distances = Float64[]
    dimensions = (n -> n.dimension).(nodes)
    delta_x = [Vector{Float64}(undef, d) for d in dimensions]
    while true
        push!(distances, distance_from_consensus(nodes))
        laplacian_step!(nodes, delta_x)

        if distances[end] <= epsilon
            break
        end
    end
    push!(distances, distance_from_consensus(nodes))
    return distances
end

function iterate_laplacian!(nodes::Vector{<:AbstractSheafNode}, epsilon::Float64, clusters::Vector{Vector{Int}})
    distances = Float64[]
    dimensions = (n -> n.dimension).(nodes)
    delta_x = [Vector{Float64}(undef, d) for d in dimensions]
    while true
        push!(distances, distance_from_consensus(nodes, clusters))
        laplacian_step!(nodes, delta_x, clusters)

        if distances[end] <= epsilon
            break
        end
    end
    push!(distances, distance_from_consensus(nodes, clusters))
    return distances
end


# Randomly reinitialize the nodes states
function random_initialization(nodes::Vector{<:AbstractSheafNode})
    for node in nodes
        x = rand(node.dimension)
        node.x = x
        for (n, rm) in node.neighbors
            take!(node.out_channels[n])
            put!(node.out_channels[n], rm * x)
        end
    end
end

function initialize!(nodes::Vector{ThreadedSheafNode}, xs)
    for (x, node) in zip(xs, nodes)
        node.x = x

        for (n, rm) in node.neighbors
            take!(node.out_channels[n])
            put!(node.out_channels[n], rm * x)
        end
    end
end

# # Construct a list of ThreadedSheafNodes from a MatrixSheaf
# function threaded_sheaf(s::MatrixSheaf)

#     nodes = ThreadedSheafNode[]

#     for i in 1:blocksize(s.x)[1]
#         push!(nodes, ThreadedSheafNode(i, size(s.x[Block(i, 1)])[1],
#             Dict{Int64,SparseMatrixCSC{Float64,Int64}}(),
#             Dict{Int64,Channel}(),
#             Dict{Int64,Channel}(), vec(s.x[Block(i, 1)])))
#     end


#     # Iterate edges and connect the vertices along each edge

#     for e in 1:blocksize(s.restriction_maps)[1]   # Iterate the block rows
#         i = -1
#         j = -1
#         for v in 1:blocksize(s.restriction_maps)[2]
#             if i == -1 && !(iszero(s.restriction_maps[Block(e, v)]))
#                 i = v
#             elseif !(iszero(s.restriction_maps[Block(e, v)]))
#                 j = v
#                 break
#             end
#         end
#         A = s.restriction_maps[Block(e, i)]
#         B = s.restriction_maps[Block(e, j)]

#         nodes[i].neighbors[j] = A
#         nodes[j].neighbors[i] = B

#         i_to_j_channel = Channel{Vector{Float64}}(2)
#         j_to_i_channel = Channel{Vector{Float64}}(2)

#         nodes[i].in_channels[j] = j_to_i_channel
#         nodes[i].out_channels[j] = i_to_j_channel
#         put!(i_to_j_channel, A * nodes[i].x)

#         nodes[j].in_channels[i] = i_to_j_channel
#         nodes[j].out_channels[i] = j_to_i_channel
#         put!(j_to_i_channel, B * nodes[j].x)
#     end

#     return nodes
# end


function random_async_threaded_sheaf(num_nodes, edge_probability, restriction_map_dimension, restriction_map_density, max_communication_delay)
    nodes = AsyncSheafNode[]
    coin()::Bool = rand() < edge_probability
    n, p, B = restriction_map_dimension, restriction_map_density, max_communication_delay
        period = rand(1:B)
        println("Everyone has the same period: $period")
        phase = rand(0:period-1)
    for i in 1:num_nodes

        push!(nodes, AsyncSheafNode(i, n,
            Dict{Int64,SparseMatrixCSC{Float64,Int64}}(),
            Dict{Int64,Channel}(),
            Dict{Int64,Channel}(), rand(n), 
            period, phase, 0)
            )
    end

    for i in 1:num_nodes
        for j in i+1:num_nodes
            if coin()
                A = sprand(n, n, p)
                B = sprand(n, n, p)

                nodes[i].neighbors[j] = A
                nodes[j].neighbors[i] = B

                i_to_j_channel = Channel{Vector{Float64}}(2)
                j_to_i_channel = Channel{Vector{Float64}}(2)

                nodes[i].in_channels[j] = j_to_i_channel
                nodes[i].out_channels[j] = i_to_j_channel
                put!(i_to_j_channel, A * nodes[i].x)

                nodes[j].in_channels[i] = i_to_j_channel
                nodes[j].out_channels[i] = j_to_i_channel
                put!(j_to_i_channel, B * nodes[j].x)
            end
        end
    end
    return nodes
end



# Pass in a function to make a certain type of restriction map?
function random_async_threaded_sheaf(g::Graph, restriction_map_dimension, restriction_map_density, max_communication_delay)
    nodes = AsyncSheafNode[]
    n, p, B = restriction_map_dimension, restriction_map_density, max_communication_delay

        period = rand(1:B)
        phase = rand(0:period-1)
    for i in 1:nv(g)

        push!(nodes, AsyncSheafNode(i, n,
            Dict{Int64,SparseMatrixCSC{Float64,Int64}}(),
            Dict{Int64,Channel}(),
            Dict{Int64,Channel}(), rand(n), period, phase, 0))
    end

    for e in edges(g)
        A = sprand(n, n, p)
        B = sprand(n, n, p)
        # A = I(n)
        # B = I(n)

        i, j = src(e), dst(e)

        nodes[i].neighbors[j] = A
        nodes[j].neighbors[i] = B

        i_to_j_channel = Channel{Vector{Float64}}(2)
        j_to_i_channel = Channel{Vector{Float64}}(2)

        nodes[i].in_channels[j] = j_to_i_channel
        nodes[i].out_channels[j] = i_to_j_channel
        put!(i_to_j_channel, A * nodes[i].x)

        nodes[j].in_channels[i] = i_to_j_channel
        nodes[j].out_channels[i] = j_to_i_channel
        put!(j_to_i_channel, B * nodes[j].x)
    end

    return nodes
end


# More generic version where you can pass in a function to generate the restriction maps
# Map generator should take in dimension and density (density can be nothing)

function random_async_threaded_sheaf(
    g::Graph,
    restriction_map_dimension;
    restriction_map_density=nothing,
    max_communication_delay=1,
    map_generator=(n, p=nothing) -> I(n)
)
    println("Using map generator: $map_generator")
    nodes = AsyncSheafNode[]
    n = restriction_map_dimension
    p = restriction_map_density
    B = max_communication_delay

    for i in 1:nv(g)
        period = rand(1:B)
        phase = rand(0:period-1)
        push!(nodes, AsyncSheafNode(i, n,
            Dict{Int64,SparseMatrixCSC{Float64,Int64}}(),
            Dict{Int64,Channel}(),
            Dict{Int64,Channel}(), rand(n), period, phase, 0))
    end

    for e in edges(g)
        i, j = src(e), dst(e)
        A = map_generator(n, p)
        Bmat = map_generator(n, p)

        nodes[i].neighbors[j] = A
        nodes[j].neighbors[i] = Bmat

        i_to_j_channel = Channel{Vector{Float64}}(2)
        j_to_i_channel = Channel{Vector{Float64}}(2)

        nodes[i].in_channels[j] = j_to_i_channel
        nodes[i].out_channels[j] = i_to_j_channel
        put!(i_to_j_channel, A * nodes[i].x)

        nodes[j].in_channels[i] = i_to_j_channel
        nodes[j].out_channels[i] = j_to_i_channel
        put!(j_to_i_channel, Bmat * nodes[j].x)
    end

    return nodes
end


function local_laplacian_step!(node::AsyncSheafNode, step_size)
    x_old = node.x
    delta_x = zeros(node.dimension)
    node.iteration += 1

    # Update every iteration
    # if rand() < prob_update
        for (n, rm) in node.neighbors
            outgoing_edge_val = rm * x_old
            incoming_edge_val = fetch(node.in_channels[n])
            delta_x -= rm' * (outgoing_edge_val - incoming_edge_val)
        end
        x_new = x_old + step_size * delta_x
        node.x = x_new
    # end

    # Broadcast every phase
    if node.iteration % node.period == node.phase
        # println("Node $(node.id) broadcasting at iteration $(node.iteration)")
        for (n, rm) in node.neighbors
            take!(node.out_channels[n])
            put!(node.out_channels[n], rm * node.x)
        end
    end
end

# Note:  Assumes all vertex and edge stalks have the same dimension
function coboundary_map(nodes::Vector{<:AbstractSheafNode})
    nverts = length(nodes)
    d = nodes[1].dimension
    # Count edges and assign edge indices
    edge_tuples = Tuple{Int,Int}[]
    for i in 1:nverts
        for (j, _) in nodes[i].neighbors
            if i < j
                push!(edge_tuples, (i, j))
            end
        end
    end
    nedges = length(edge_tuples)
    δ = zeros(Float64, nedges*d, nverts*d)
    edge_idx = 1
    for (i, j) in edge_tuples
        A = nodes[i].neighbors[j]
        B = nodes[j].neighbors[i]
        # Place A in δ[edge, i]
        δ[(edge_idx-1)*d+1:edge_idx*d, (i-1)*d+1:i*d] .= A
        # Place -B in δ[edge, j]
        δ[(edge_idx-1)*d+1:edge_idx*d, (j-1)*d+1:j*d] .= -B
        edge_idx += 1
    end
    return δ
end


function lipschitz_constant(nodes::Vector{<:AbstractSheafNode})
    δ = coboundary_map(nodes)
    L = δ' * δ
    K = opnorm(Matrix(L))
    return K
end





# function laplacian_step_async!(nodes::Vector{<:AbstractSheafNode}, step_size::Float64, prob_update::Float64, prob_broadcast::Float64)
#     Threads.@threads for node in nodes
#         local_laplacian_step_async!(node, step_size, prob_update, prob_broadcast)
#     end
# end

# function iterate_laplacian_async!(nodes::Vector{<:AbstractSheafNode}, step_size, epsilon::Float64, prob_update::Float64, prob_broadcast::Float64)
#     distances = Float64[]

#     while true
#         push!(distances, distance_from_consensus(nodes))
#         laplacian_step_async!(nodes, step_size, prob_update, prob_broadcast)

#         if distances[end] <= epsilon
#             break
#         end
#     end
#     push!(distances, distance_from_consensus(nodes))
#     return distances
# end

# # Returns a list of distances from consensus over the iterations
# function iterate_laplacian_async!(nodes::Vector{<:AbstractSheafNode}, step_size, num_iters::Int, prob_update::Float64, prob_broadcast::Float64)
#     distances = Float64[]

#     for _ in 1:num_iters
#         push!(distances, distance_from_consensus(nodes))
#         laplacian_step_async!(nodes, step_size, prob_update, prob_broadcast)
#     end
#     push!(distances, distance_from_consensus(nodes))
#     return distances
# end


end