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


# GR plotting backend sometimes fails to precompile due to missing LERC_jll dependency
# The following code can be used to fix this issue manually

# using Pkg
# # skip activation of a temp environment if you are trying to fix your environment
# Pkg.activate(; temp=true)
# # Add LERC_jll version 3
# Pkg.add([
#     PackageSpec(name="GR"),
#     PackageSpec(name="LERC_jll", version="3")
# ])
# # Force GR to precompile
# Base.compilecache(Base.PkgId(Base.UUID("28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"), "GR"))