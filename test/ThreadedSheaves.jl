using AlgebraicOptimization
using BlockArrays
using Test
using Plots
using Graphs
using LinearAlgebra
using Random




identity_map(n, p=nothing) = I(n)
dense_random_map(n, p=nothing) = rand(Float64, n, n)
orthogonal_random_map(n, p=nothing) = Matrix(qr(randn(n, n)).Q)
sparse_random_map(n, p) = sprand(n, n, p)





# Construct a random threaded sheaf from a graph

# Start by constructing a complete graph on n vertices
n = 8
g = complete_graph(n)
B = 1

nodes_id = random_async_threaded_sheaf(g, 2; map_generator=orthogonal_random_map)
ThreadedSheaves.coboundary_map(nodes_id)

# Calculate step sizes
K = lipschitz_constant(nodes_id)
step_size = Float64(.99 * (2 / (K * (1 + 2 * sqrt(n) * B))))






# If this doesn't converge, we are screwed
# Basic case: Identity maps and all nodes initialized to the same value
B = 1
connected_graph = complete_graph(8)
nodes = random_async_threaded_sheaf(connected_graph, 1, 1.0, B
)
ThreadedSheaves.coboundary_map(nodes)
for node in nodes
    node.x .= rand(Float64)  # Initialize all nodes to the same random value
    for (n, rm) in node.neighbors
        take!(node.out_channels[n])  # Remove old value
        put!(node.out_channels[n], rm * node.x)  # Insert new value
    end
end

# @test distance_from_consensus(nodes) == 0 # Should be 0

loss = iterate_laplacian!(nodes, .001, 10000)

# Print all of the node values
for node in nodes
    println("Node $(node.id): ", node.x)
end

# Plot results
iters = 1:length(loss)
plot(iters, loss, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss vs Iteration", label="Identity Sheaf")

distance_from_consensus(nodes)







# # Sync version of laplacian iteration on threaded sheaf nodes
# nodes = random_threaded_sheaf(8, 0.3, 10, 0.3)
# random_initialization(nodes)

# convergence_threshold = 1.0
# step_size = .002
# loss = iterate_laplacian!(nodes, step_size, 100)

# loss = iterate_laplacian!(nodes, step_size, convergence_threshold)
# @test loss[end] < convergence_threshold 


# Why are we sometimes getting super sharp dropoffs at the beginning?

# Why is there no convergence at all in the faulty test case where all the nodes had the same giant period in the "off seasons"?



# Seed random
Random.seed!(42)
# Async version of laplacian iteration on threaded sheaf nodes. phase <= period <= B.
N = 10
B = 100
num_iters = 50000
nodes = random_async_threaded_sheaf(N, 0.3, 10, 0.3, B)

# Calculate step sizes
K = lipschitz_constant(nodes)
step_size = Float64(.99 * (2 / (K * (1 + 2 * sqrt(N) * B))))
step_size_divergence = Float64(1 / K)

# Initialize nodes
random_initialization(nodes)
nodes_divergence = deepcopy(nodes)
nodes_sync = deepcopy(nodes)

for n in nodes_sync  # Set all nodes to be perfectly synchronized
    n.period = 1
    n.phase = 0
end
nodes_sync_divergence = deepcopy(nodes_sync)

# Run iterations
loss = iterate_laplacian!(nodes, step_size, num_iters)
loss_divergence = iterate_laplacian!(nodes_divergence, step_size_divergence, num_iters)
loss_sync = iterate_laplacian!(nodes_sync, step_size, num_iters)
loss_sync_divergence = iterate_laplacian!(nodes_sync_divergence, step_size_divergence, num_iters)

# Plot results
iters = 1:length(loss)
plot(iters, loss, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss vs Iteration", label="Async Sheaf")
plot!(iters, loss_divergence, yscale=:log10, label="Async Sheaf 1/K Step")
plot!(iters, loss_sync, yscale=:log10, label="Sync Sheaf")
plot!(iters, loss_sync_divergence, yscale=:log10, label="Sync Sheaf 1/K Step")









# Several iterations and plot average result

num_trials = 10
max_iters = 100000
avg_loss = zeros(max_iters + 1)


for trial in 1:num_trials
    nodes = random_async_threaded_sheaf(8, 0.3, 10, 0.3, 100)
    random_initialization(nodes)
    loss = iterate_laplacian!(nodes, step_size, max_iters)
    avg_loss .+= loss
end

avg_loss ./= num_trials
iters = 1:length(avg_loss)
# plot(iters, 1 ./ avg_loss, xlabel="Iteration", ylabel="Average 1 / Loss (10 trials)", title="Average Inverse Loss vs Iteration", legend=false)
plot(iters, loss, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss vs Iteration", legend=false)



# # Infrequent updates, frequent broadcasts
# num_trials = 10
# max_iters = 100000
# avg_loss = zeros(max_iters + 1)


# prob_update = 0.001
# prob_broadcast = 0.8

# for trial in 1:num_trials
#     nodes = random_threaded_sheaf(8, 0.3, 10, 0.3)
#     random_initialization(nodes)
#     loss = iterate_laplacian_async!(nodes, step_size, max_iters, prob_update, prob_broadcast)
#     avg_loss .+= loss
# end

# avg_loss ./= num_trials
# iters = 1:length(avg_loss)
# plot(iters, 1 ./ avg_loss, xlabel="Iteration", ylabel="Average 1 / Loss ($num_trials trials)", title="Average Inverse Loss vs Iteration", legend=false)

# plot(iters, loss, yscale=:log10, xlabel="Iteration", ylabel="Loss", title="Loss vs Iteration", legend=false)

# loss = iterate_laplacian_async!(nodes, step_size, convergence_threshold, prob_update, prob_broadcast)
# @test loss[end] < convergence_threshold 

# # Converting a random MatrixSheaf to an array of ThreadedSheaves

# random_matrix_s = random_matrix_sheaf(10, 4, 5)
# random_threaded = threaded_sheaf(random_matrix_s)
# @test size(random_threaded)[1] == 10  # Check number of vertices matches
# @test random_threaded[1].x == vec(random_matrix_s.x[BlockArrays.Block(1, 1)])  # Check that edge dimension matches

# # Converting a MatrixSheaf with nonuniform dimension to an array of ThreadedSheaves

# non_uniform_matrix_sheaf = MatrixSheaf([1, 2, 3], [1, 2])
# add_map!(non_uniform_matrix_sheaf, 1, 1, reshape([42], 1, 1))
# add_map!(non_uniform_matrix_sheaf, 2, 1, [0 1;]) 
# add_map!(non_uniform_matrix_sheaf, 1, 2, reshape([1, 2], 2, 1))   # We're using reshape here to get Matrix types instead of vector types
# add_map!(non_uniform_matrix_sheaf, 3, 2, [5 5 5; 6 6 6]) 

# non_uniform_threaded_sheaf = threaded_sheaf(non_uniform_matrix_sheaf)

# @test size(non_uniform_threaded_sheaf)[1] == 3  # Check number of vertices matches
# @test non_uniform_threaded_sheaf[2].x == vec(non_uniform_matrix_sheaf.x[BlockArrays.Block(2, 1)])  # Check that edge dimension matches


# Simulate the same laplacian on a MatrixSheaf and a list of ThreadedSheafNodes

# random_matrix_s = random_matrix_sheaf(10, 4, 5)
# random_threaded = threaded_sheaf(random_matrix_s)

# iters = 100
# my_step_size::Float64 = .01

# for _ in 1:iters
#     laplacian_update!(random_matrix_s, my_step_size)
# end

# iterate_laplacian!(random_threaded, my_step_size, iters)

# TODO: Add a test for equality of the x values