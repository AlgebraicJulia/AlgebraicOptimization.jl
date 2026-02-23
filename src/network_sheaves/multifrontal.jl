using AlgebraicOptimization
using Graphs

function sheaf_from_graph(g::Graph, stalk_dim::Int, rm_generator::Function)
    n = nv(g)
    s = EuclideanSheaf{Float64}(repeat([stalk_dim], n))

    for e in edges(g)
        i, j = src(e), dst(e)
        rm1 = rm_generator(stalk_dim)
        rm2 = rm_generator(stalk_dim)
        add_sheaf_edge!(s, i, j, rm1, rm2)
    end
    return s
end

rm_generator(vertex_stalk_dim, edge_stalk_dim) = rand(edge_stalk_dim, vertex_stalk_dim)

N = 50
p = 0.1
g = erdos_renyi(N, p)

s = sheaf_from_graph(g, 10, v -> rm_generator(v, 5))

L = sheaf_laplacian_matrix(s)