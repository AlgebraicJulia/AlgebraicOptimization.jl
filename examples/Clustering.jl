using AlgebraicOptimization
using Graphs
using Plots
using MatrixMarket
using SparseArrays
using SuiteSparseMatrixCollection

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

nodes = random_threaded_sheaf(g, 5, 0.3)

clusters = compute_clusters(g, 4)

loss = iterate_laplacian!(nodes, 10000, clusters)

plot(loss)