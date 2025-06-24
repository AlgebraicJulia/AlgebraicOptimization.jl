using AlgebraicOptimization
using Graphs
using SparseArrays
using LinearAlgebra
using Krylov



N = 200

bump(x, mu=0, sigma=10) = begin
    z = exp.(-(x .- mu) .^ 2 ./ sigma)
    z /= sum(z)
end

#mesh = path_graph(N)
bump(-3:3, 0, 5)
#big_L = Graphs.LinAlg.laplacian_matrix(mesh, Float64)

function line_laplacian(n)
    d = vcat([1.0], 2 * ones(n - 2), [1.0])
    return Tridiagonal(-ones(n - 1), d, -ones(n - 1))
end

L = line_laplacian(N)

#M = Diagonal(diag(big_L))
bumpdomain = -6:6
big_b = zeros(N)
big_b[12 .+ bumpdomain] = -1.0 * bump(bumpdomain, 0, 1)
big_b[40 .+ bumpdomain] = 1.0 * bump(bumpdomain, 0, 1)
big_b[111 .+ bumpdomain] = -1.0 * bump(bumpdomain, 0, 1)
big_b[170 .+ bumpdomain] = 1.0 * 1.0 * bump(bumpdomain, 0, 1)

plot(big_b)
#big_b = big_b .- (sum(big_b) / length(big_b))


# solves Lx=b for x
x, stats = cr(L, big_b; history=true)

#factors = qr(collect(big_L); pivot=true)


Nhalf = ceil(Int, N / 2)
AB = 10
A = Nhalf + AB
B = Nhalf + AB

s = CellularSheaf([A, B], [AB])

L1 = line_laplacian(A)
L2 = line_laplacian(B)

b1 = big_b[1:A]
b2 = big_b[end-B+1:end]

f1(x) = norm(L1 * x - b1)^2
f2(x) = norm(L2 * x - b2)^2

p1 = zeros(AB, A)
p1[1:AB, end-AB+1:end] .= I(AB)
p1

p2 = zeros(AB, B)
p2[1:AB, 1:AB] = I(AB)
p2

set_edge_maps!(s, 1, 2, 1, p1, p2)

hp = HomologicalProgram([f1, f2], s, zeros(AB))


primal_sol, dual_sol = solve(hp, ADMM(2.0, 100))

x1 = primal_sol[Block(1)]
x2 = primal_sol[Block(2)]


x_sol = vcat(x1[1:end-AB], x2)

p = plot(x_sol / norm(x_sol), label="hp")
plot!(p, x / norm(x), color=:teal, label="krylov")
plot!(p, big_b / norm(big_b), color=:purple, label="b")
p