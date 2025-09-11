using AlgebraicOptimization
using BlockArrays
using Test
using Plots


# Sync version of laplacian iteration on threaded sheaf nodes
nodes = random_threaded_sheaf(8, 0.3, 10, 0.3)
random_initialization(nodes)

convergence_threshold = 1.0
step_size = 2.0f-2 
loss = iterate_laplacian!(nodes, step_size, 100)

loss = iterate_laplacian!(nodes, step_size, convergence_threshold)
@test loss[end] < convergence_threshold 




# Async version of laplacian iteration on threaded sheaf nodes. phase <= period <= B.
N = 8
B = 1000
nodes = random_async_threaded_sheaf(N, 0.3, 10, 0.3, B)
K = lipschitz_constant(nodes)

step_size = 2 / (K * (1 + 2 * sqrt(N) * B))


random_initialization(nodes)

convergence_threshold = 1.0
step_size = 2.0f-2 
prob_update = 0.8
prob_broadcast = 0.1
loss = iterate_laplacian!(nodes, step_size, 100000)

iters = 1:length(loss)
plot(iters, 1 ./ loss, xlabel="Iteration", ylabel="1 / Loss", title="Inverse Loss vs Iteration", legend=false)




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
# my_step_size::Float32 = .01

# for _ in 1:iters
#     laplacian_update!(random_matrix_s, my_step_size)
# end

# iterate_laplacian!(random_threaded, my_step_size, iters)

# TODO: Add a test for equality of the x values