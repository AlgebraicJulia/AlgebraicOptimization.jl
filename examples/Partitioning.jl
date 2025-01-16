# An example of how to use METIS to partition a cellular sheaf, then run the sheaf laplacian dynamics on 
# multiple threads.

using AlgebraicOptimization
using MatrixMarket
using SparseArrays
using SuiteSparseMatrixCollection
using Graphs

# construct the database
# http://sparse.tamu.edu
ssmc = ssmc_db()

# the name of the graph to fetch
name = "fe_tooth"

# fetch the graph
graph = mmread(joinpath(fetch_ssmc(ssmc[ssmc.name.==name, :], format="MM")[1], "$(name).mtx"))

# remove self edges
fkeep!((i, j, v) -> i != j, graph)

# remove weights
fill!(nonzeros(graph), 1)