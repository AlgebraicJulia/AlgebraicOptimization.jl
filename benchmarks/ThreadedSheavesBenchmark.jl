include("../src/ThreadedSheaves.jl")

# Strip all the @threads macros out of the functions.
# I know this isn't single-threaded optimal, but this is just
# to get an idea of speedup.


function st_descent_direction!(nodes, results::Vector{Vector{Float32}})
    #println("Computing a descent direction")
    # Allocate a shared memory array for results
    #dimensions = (n -> n.dimension).nodes
    #results = [Vector{Float32}(undef, d) for d in dimensions] # make a version that passes this in as an argument to override

    for i in eachindex(nodes)
        results[i] = local_descent_direction(nodes[i])
    end
end


function st_consensus_objective(nodes::Vector{ThreadedSheafNode}, xs::Vector{Vector{Float32}})
    #println("Computing objective")
    losses = 0.0
    for i in eachindex(nodes)
        losses += local_consensus_objective(nodes[i], xs[i])
    end
    return losses
end

function st_sum(x, y)
    #println("Computing a sum")
    return [x + y for (x,y) in zip(x,y)]
end

function st_line_search(nodes, delta_x::Vector{Vector{Float32}})
    #println("Got to line search")
    @assert length(nodes) == length(delta_x)
    c = 0.5
    τ = 0.1
    m = 0.0
    for i in eachindex(delta_x)
        m += (-delta_x[i])'*delta_x[i]
    end
    #println(m)
    t = -c*m
    a=Float32(0.001)
    x = [node.x for node in nodes]
    while st_consensus_objective(nodes, x) - st_consensus_objective(nodes, st_sum(x, a .* delta_x)) < a*t
        a = τ*a
    end
    #println("Step size: $a")
    return a
end

function st_update_nodes!(nodes, a, delta_x)
    #println("Got to update")
    for i in eachindex(nodes)
        nodes[i].x += a*delta_x[i]
        for (n, rm) in nodes[i].neighbors
            # Consume from buffers
            take!(nodes[i].in_channels[n])
            put!(nodes[i].out_channels[n], rm*nodes[i].x)
        end
    end
end

function st_laplacian_step!(nodes, delta_x::Vector{Vector{Float32}})
    #println("Taking a laplacian step")
    st_descent_direction!(nodes, delta_x)
    step_size = st_line_search(nodes, delta_x)
    st_update_nodes!(nodes, step_size, delta_x)
end

function st_iterate_laplacian!(nodes, num_iters::Int32)
    distances = Float64[]
    dimensions = (n -> n.dimension).(nodes)
    delta_x = [Vector{Float32}(undef, d) for d in dimensions]
    for _ in 1:num_iters
        push!(distances, distance_from_consensus(nodes))
        st_laplacian_step!(nodes, delta_x)
    end
    push!(distances, distance_from_consensus(nodes))
    return distances
end

nodes = random_threaded_sheaf(nthreads(), .5, 100, .3)

dimensions = (n -> n.dimension).(nodes)
x0 = [rand(d) for d in dimensions]

initialize!(nodes, x0)

st_iterate_laplacian!(nodes, 100)

initialize!(nodes, x0)

t1 = @time st_iterate_laplacian!(nodes, 1000)

initialize!(nodes, x0)

iterate_laplacian!(nodes, 100)

initialize!(nodes, x0)

t2 = @time iterate_laplacian!(nodes, 1000)