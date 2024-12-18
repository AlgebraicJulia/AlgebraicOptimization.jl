include("SheafNodes.jl")
using .SheafNodes
using Base.Threads
using SparseArrays
using LinearAlgebra

function local_laplacian_step!(node, step_size)
    x_old = node.x
    delta_x = zeros(node.dimension)

    for (n, rm) in node.neighbors
        outgoing_edge_val = rm*x_old
        incoming_edge_val = take!(node.in_channels[n])
        delta_x += rm'*(outgoing_edge_val - incoming_edge_val)
    end
    x_new = x_old - step_size*delta_x

    for (n, rm) in node.neighbors
        put!(node.out_channels[n], rm*x_new)
    end

    node.x = x_new
end

function laplacian_step!(nodes, step_size::Float32)
    Threads.@threads for node in nodes
        local_laplacian_step!(node, step_size)
    end
end

function random_threaded_sheaf(num_nodes, edge_probability, restriction_map_dimension, restriction_map_density)
    nodes = ThreadedSheafNode[]
    coin()::Bool = rand() < edge_probability
    n, p = restriction_map_dimension, restriction_map_density
    for i in 1:num_nodes
        push!(nodes, ThreadedSheafNode(i, n,
            Dict{Int32, SparseMatrixCSC{Float32, Int32}}(),
            Dict{Int32, Channel}(),
            Dict{Int32, Channel}(), rand(n)))
    end

    for i in 1:num_nodes
        for j in i+1:num_nodes
            if coin()
                A = sprand(n,n,p)
                B = sprand(n,n,p)

                nodes[i].neighbors[j] = A
                nodes[j].neighbors[i] = B

                i_to_j_channel = Channel{Vector{Float32}}(1)
                j_to_i_channel = Channel{Vector{Float32}}(1)

                nodes[i].in_channels[j] = j_to_i_channel
                nodes[i].out_channels[j] = i_to_j_channel
                put!(i_to_j_channel, A*nodes[i].x)

                nodes[j].in_channels[i] = i_to_j_channel
                nodes[j].out_channels[i] = j_to_i_channel
                put!(j_to_i_channel, B*nodes[j].x)
            end
        end
    end
    return nodes
end


function distance_from_consensus(nodes)
    total_distance = 0.0
    for node in nodes
        node_distance = 0.0
        # There is some double counting happening in here but idrc
        for ((_, in_channel), (_, out_channel)) in zip(node.in_channels, node.out_channels)
            node_distance += norm(fetch(in_channel) - fetch(out_channel))
        end
        total_distance += node_distance
    end
    return total_distance
end

# Returns a list of distances from consensus over the iterations
function iterate_laplacian!(nodes, step_size, num_iters)
    distances = Float64[]

    for _ in 1:num_iters+1
        laplacian_step!(nodes, step_size)
        push!(distances, distance_from_consensus(nodes))
    end
    return distances
end

# Randomly reinitialize the nodes states
function random_initialization(nodes)
    for node in nodes
        node.x = rand(node.dimension)
    end
end