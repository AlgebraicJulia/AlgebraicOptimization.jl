module DistributedSheaves

export random_distributed_sheaf, iterate_laplacian!, distance_from_consensus

using Distributed
using LinearAlgebra

function laplacian_step!(workers::Vector{Int}, node_refs::Vector{Future}, step_size::Float32)
    # Define this function for all the nodes
    @everywhere function local_laplacian_step!(node_ref, step_size)
        x_old = fetch(node_ref).x
        # println("current x: $x_old")
        delta_x = zeros(fetch(node_ref).dimension)

        # Compute updated state
        for (n, rm) in fetch(node_ref).neighbors
            outgoing_edge_val = rm * x_old
            incoming_edge_val = take!(fetch(node_ref).in_channels[n])
            delta_x += rm' * (outgoing_edge_val - incoming_edge_val)
        end
        x_new = x_old - step_size * delta_x

        # Broadcast updated state to neighbors
        for (n, rm) in fetch(node_ref).neighbors
            put!(fetch(node_ref).out_channels[n], rm * x_new)
        end

        # Update local state
        fetch(node_ref).x = x_new
    end # local_laplacian_step

    # Have each node execute a local laplacian step
    for (w, nr) in zip(workers, node_refs)
        remote_do(local_laplacian_step!, w, nr, step_size)
    end
end

# Returns remote references for each node and communication channels for each edge
function random_distributed_sheaf(num_nodes, edge_probability, restriction_map_dimension, restriction_map_density)
    # Set up workers and make sure they have all the code they need
    workers = addprocs(num_nodes)
    @everywhere @eval using SparseArrays
    @everywhere include("src/SheafNodes.jl")
    @everywhere @eval using .SheafNodes
    n, p = restriction_map_dimension, restriction_map_density
    coin()::Bool = rand() < edge_probability

    # Spawn sheaf nodes on every worker
    node_refs = Future[]
    for w in workers
        push!(node_refs, @spawnat w DistributedSheafNode(w, n,
            Dict{Int32,SparseMatrixCSC{Float32,Int32}}(), # restriction maps
            #Dict{Int32, Matrix{Float32}}(),
            Dict{Int32,RemoteChannel}(),                   # inbound channels
            Dict{Int32,RemoteChannel}(), rand(n)))         # outbound channels, initial state
    end

    # Add random sheaf edges
    for i in 2:num_nodes+1 # worker nodes start at index 2 because the main node is index 1!
        for j in i+1:num_nodes+1
            if coin() # add an edge between i and j
                # This requires two restriction maps:
                # i -- A --> e <-- B -- j
                # A should live on proc i and B should live on proc j
                Aref = @spawnat i sprand(n, n, p)
                Bref = @spawnat j sprand(n, n, p)
                #Aref = @spawnat i rand(n,n)
                #Bref = @spawnat j rand(n,n)

                remote_do(node_ref -> fetch(node_ref).neighbors[j] = fetch(Aref), i, node_refs[i-1])
                remote_do(node_ref -> fetch(node_ref).neighbors[i] = fetch(Bref), j, node_refs[j-1])

                # Make remote channels for these nodes to communicate
                i_to_j_channel = RemoteChannel(() -> Channel{Vector{Float32}}(1))
                j_to_i_channel = RemoteChannel(() -> Channel{Vector{Float32}}(1))

                # Set the appropriate remote channels for each node and seed them with the initial values
                remote_do(node_ref -> begin
                        fetch(node_ref).in_channels[j] = j_to_i_channel
                        fetch(node_ref).out_channels[j] = i_to_j_channel
                        put!(i_to_j_channel, fetch(Aref) * fetch(node_ref).x)
                    end, i, node_refs[i-1])
                remote_do(node_ref -> begin
                        fetch(node_ref).in_channels[i] = i_to_j_channel
                        fetch(node_ref).out_channels[i] = j_to_i_channel
                        put!(j_to_i_channel, fetch(Bref) * fetch(node_ref).x)
                    end, j, node_refs[j-1])
            end
        end
    end
    return workers, node_refs
end

function distance_from_consensus(node_refs::Vector{Future})
    @everywhere @eval using LinearAlgebra
    return @distributed (+) for nr in node_refs
        node_distance = 0.0
        # There is some double counting happening in here but idrc
        for ((_, in_channel), (_, out_channel)) in zip(fetch(nr).in_channels, fetch(nr).out_channels)
            node_distance += norm(fetch(in_channel) - fetch(out_channel))
        end
        node_distance
    end
end

# Returns a list of distances from consensus over the iterations
function iterate_laplacian!(workers, node_refs, step_size, num_iters)
    distances = Float64[]

    for _ in 1:num_iters+1
        laplacian_step!(workers, node_refs, step_size)
        push!(distances, distance_from_consensus(node_refs))
    end
    return distances
end

end

# Some tests
# TODO: Figure out why this blows up for larger examples
# For some reason these tests error if uncommenting because of some SheafNode load bug
# Until we figure that out, just do these commands in the REPL to see the results.
#=using Plots

workers, node_refs = random_distributed_sheaf(5, .5, 5, .5)

loss = iterate_laplacian!(workers, node_refs, Float32(0.1), 20)

plot(loss)=#