using AlgebraicOptimization
using Test


nodes = random_threaded_sheaf(8, 0.3, 10, 0.3)
random_initialization(nodes)

convergence_threshold = 1.0

loss = iterate_laplacian!(nodes, convergence_threshold)

@test loss[end] < convergence_threshold
