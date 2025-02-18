using AlgebraicOptimization
using Graphs
using Plots
using MatrixMarket
using SparseArrays
using SuiteSparseMatrixCollection
using BenchmarkTools

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

#g = erdos_renyi(16, 0.3)

setup_times = Float64[]
cluster_times = Float64[]
laplacian_times = Float64[] 

for i in 1:20

    push!(setup_times, @elapsed (nodes = random_threaded_sheaf(g, 50, 0.2)))

    push!(cluster_times, @elapsed(clusters = compute_clusters(g, i)))

    push!(laplacian_times,  (@elapsed iterate_laplacian!(nodes, 10, clusters)))

end

p = plot(
    1:length(setup_times), setup_times, label="Setup Times", 
    xlabel="Clusters", ylabel="Time (s)", title="Benchmark Times"
)
plot!(1:length(laplacian_times), cluster_times, label="Clustering Times")
plot!(1:length(laplacian_times), laplacian_times, label="Laplacian Times")


savefig(p, "./examples/clustering_benchmark.png")
