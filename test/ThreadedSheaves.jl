using AlgebraicOptimization
using BlockArrays
using Test


nodes = random_threaded_sheaf(8, 0.3, 10, 0.3)
random_initialization(nodes)

convergence_threshold = 1.0

loss = iterate_laplacian!(nodes, convergence_threshold)

@test loss[end] < convergence_threshold


# Converting a random MatrixSheaf to an array of ThreadedSheaves

random_matrix_s = random_matrix_sheaf(10, 4, 5)
random_threaded = threaded_sheaf(random_matrix_s)
@test size(random_threaded)[1] == 10  # Check number of vertices matches
@test random_threaded[1].x == vec(random_matrix_s.x[BlockArrays.Block(1, 1)])  # Check that edge dimension matches

# Converting a MatrixSheaf with nonuniform dimension to an array of ThreadedSheaves

non_uniform_matrix_sheaf = MatrixSheaf([1, 2, 3], [1, 2])
add_map!(non_uniform_matrix_sheaf, 1, 1, reshape([42], 1, 1))
add_map!(non_uniform_matrix_sheaf, 2, 1, [0 1;]) 
add_map!(non_uniform_matrix_sheaf, 1, 2, reshape([1, 2], 2, 1))   # We're using reshape here to get Matrix types instead of vector types
add_map!(non_uniform_matrix_sheaf, 3, 2, [5 5 5; 6 6 6]) 

non_uniform_threaded_sheaf = threaded_sheaf(non_uniform_matrix_sheaf)

@test size(non_uniform_threaded_sheaf)[1] == 3  # Check number of vertices matches
@test non_uniform_threaded_sheaf[2].x == vec(non_uniform_matrix_sheaf.x[BlockArrays.Block(2, 1)])  # Check that edge dimension matches


