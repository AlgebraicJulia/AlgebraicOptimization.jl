using Distributed
#using ..SheafNodes

function laplacian_step(workers::Vector{Int}, node_refs::Vector{Future}, step_size::Float32)
    @everywhere function local_laplacian_step(node_ref, step_size)
        x_old = fetch(node_ref).x
        println("current x: $x_old")
        delta_x = zeros(fetch(node_ref).dimension)

        # Compute updated state
        for (n, rm) in fetch(node_ref).neighbors
            outgoing_edge_val = rm*x_old
            incoming_edge_val = take!(fetch(node_ref).in_channels[n])
            delta_x += rm'*(outgoing_edge_val - incoming_edge_val)
        end
        x_new = x_old - step_size*delta_x

        # Broadcast updated state to neighbors
        for (n, rm) in fetch(node_ref).neighbors
            put!(fetch(node_ref).out_channels[n], rm*x_new)
        end

        # Update local state
        fetch(node_ref).x = x_new
    end

    for (w, nr) in zip(workers, node_refs)
        remote_do(local_laplacian_step, w, nr, step_size)
    end
end

#=@everywhere function local_laplacian_step(node_ref)
    #node = fetch(node_ref)
    x_old = fetch(node_ref).x
    println("current x: $x_old")
    x_new = zeros(fetch(node_ref).dimension)

    # Compute updated state
    for (n, rm) in fetch(node_ref).neighbors
        outgoing_edge_val = rm*x_old
        incoming_edge_val = take!(fetch(node_ref).in_channels[n])
        x_new += rm'*(outgoing_edge_val - incoming_edge_val)
    end

    # Broadcast updated state to neighbors
    for (n, rm) in fetch(node_ref).neighbors
        put!(fetch(node_ref).out_channels[n], rm*x_new)
    end

    # Update local state
    fetch(node_ref).x = x_new
end=#

# Returns remote references for each node and communication channels for each edge
function random_distributed_sheaf(num_nodes, edge_probability, restriction_map_dimension, restriction_map_density)
    workers = addprocs(num_nodes)
    @everywhere @eval using SparseArrays
    @everywhere include("src/SheafNodes.jl")
    @everywhere @eval using .SheafNodes
    n,p = restriction_map_dimension, restriction_map_density
    coin()::Bool = rand() < edge_probability

    # Spawn sheaf nodes on every worker
    #node_refs = Dict{Int64, Future}
    node_refs = Future[]
    for w in workers
        #node_refs[w] = @spawnat w SheafNode(w, Dict{Int32, Matrix{Float32}}())
        push!(node_refs, @spawnat w SheafNode(w, n, Dict{Int32, SparseMatrixCSC{Float32, Int32}}(), Dict{Int32, RemoteChannel}(), Dict{Int32, RemoteChannel}(), rand(n)))
    end

    # Add random sheaf edges
    for i in 2:num_nodes+1 # worker nodes start at index 2 because the main node is index 1!
        for j in i+1:num_nodes+1
            if coin() # add an edge between i and j
                # This requires two restriction maps:
                # i -- A --> e <-- B -- j
                # A should live on proc i and B should live on proc j
                Aref = @spawnat i sprand(n,n,p)
                Bref = @spawnat j sprand(n,n,p)

                remote_do(node_ref -> fetch(node_ref).neighbors[j] = fetch(Aref), i, node_refs[i-1])
                remote_do(node_ref -> fetch(node_ref).neighbors[i] = fetch(Bref), j, node_refs[j-1])

                # Make remote channels for these nodes to communicate
                i_to_j_channel = RemoteChannel(() -> Channel{Vector{Float32}}(1))
                j_to_i_channel = RemoteChannel(() -> Channel{Vector{Float32}}(1))

                remote_do(node_ref -> begin 
                                        fetch(node_ref).in_channels[j] = j_to_i_channel 
                                        fetch(node_ref).out_channels[j] = i_to_j_channel
                                        put!(i_to_j_channel, fetch(Aref)*fetch(node_ref).x)
                                      end, i, node_refs[i-1])
                remote_do(node_ref -> begin 
                                        fetch(node_ref).in_channels[i] = i_to_j_channel 
                                        fetch(node_ref).out_channels[i] = j_to_i_channel
                                        put!(j_to_i_channel, fetch(Bref)*fetch(node_ref).x)
                                       end, j, node_refs[j-1])
            end
        end
    end
    return workers, node_refs
end