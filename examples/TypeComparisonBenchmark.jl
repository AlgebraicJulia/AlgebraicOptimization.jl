using AlgebraicOptimization
using Graphs
using Plots
using MatrixMarket
using SparseArrays
using SuiteSparseMatrixCollection
using BenchmarkTools
using Test

# This benchmark compares the efficiency of the laplacian update step on MatrixSheaves and ThreadedSheaves.


# construct the database
# http://sparse.tamu.edu
ssmc = ssmc_db()

# the name of the graph to fetch
#name = "fe_tooth"
name = "1138_bus"

# fetch the graph
graph = mmread(joinpath(fetch_ssmc(ssmc[ssmc.name.==name, :], format="MM")[1], "$(name).mtx"))

# remove self edges
fkeep!((i, j, v) -> i != j, graph)

# remove weights
fill!(nonzeros(graph), 1)

println(repr("text/plain", graph))
g = Graph(graph)

# Set up parameters
passes = 10
α = .01f0

# Set up sheaves
my_matrix_sheaf = random_matrix_sheaf(g, 1, 1.0)
make_coboundary!(my_matrix_sheaf)   # TODO: Potentially incorporate this into some of the constructors
my_threaded_sheaf = threaded_sheaf(my_matrix_sheaf)    # TODO: Fix bug for this constructor when there are restriction maps that are 0

# Set up benchmarks
matrix_sheaf_times = Float64[]
threaded_sheaf_times = Float64[]
dims = 1:4:10

for dim in dims
    my_matrix_sheaf = random_matrix_sheaf(g, dim, 1.0)
    make_coboundary!(my_matrix_sheaf)   # TODO: Potentially incorporate this into some of the constructors
    my_threaded_sheaf = threaded_sheaf(my_matrix_sheaf)


    # Iterate laplacian on the MatrixSheaf
    mtime = @elapsed for _ in 1:passes
        CellularSheaves.laplacian_step!(my_matrix_sheaf, α)
    end

    # Iterate laplacian on the ThreadedSheaf
    ttime = @elapsed for _ in 1:passes
        ThreadedSheaves.laplacian_step!(my_threaded_sheaf, α)  # No clustering yet
    end

    push!(matrix_sheaf_times, mtime)
    push!(threaded_sheaf_times, ttime)
end


# Check that the MatrixSheaf and ThreadedSheaf produced the same x values
my_matrix_sheaf_x = vec(vcat(my_matrix_sheaf.x))
my_threaded_sheaf_x = vcat([node.x for node in my_threaded_sheaf]...)

@test my_matrix_sheaf_x ≈ my_threaded_sheaf_x

plot(
    dims, matrix_sheaf_times, label="MatrixSheaf", 
    xlabel="Dimensions", ylabel="Time (s)", title="MatrixSheaf vs. ThreadedSheaf Times"
)
plot!(dims, threaded_sheaf_times, label="ThreadedSheaf")
