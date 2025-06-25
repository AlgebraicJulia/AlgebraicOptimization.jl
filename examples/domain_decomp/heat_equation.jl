using AlgebraicOptimization
using Graphs
using SparseArrays
using LinearAlgebra
using Krylov
using Plots
using BlockArrays

include("fnc_utils.jl")

N = 200

x, Dₓ, Dₓₓ = diffmat2(N - 1, (-1, 1))

bump(x, mu=0, sigma=10) = begin
    z = exp.(-(x .- mu) .^ 2 ./ sigma)
end


rhs_func(x) = 3bump(x, 1 / 2, 1 / 20) - 3bump(x, -1 / 2, 1 / 20)
# x, u = bvplin(1 / 20, x -> 0, x -> 0, x -> -rhs_func(x), [-1, 1], -1 / 2, -1 / 2, N)
# x, u = bvplin(1 / 20, x -> 0, x -> 0, x -> -rhs_func(x), [-1, 1], 0,0, N)

solveN = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, 1], N)
x,u = solveN(0,0)

p = plot(x, rhs_func.(x), label="b")
p = plot!(p, x, u, label="bvplin_soln")


Nhalf = ceil(Int, N / 2)
AB = 10
A = Nhalf + ceil(Int, AB / 2)
B = Nhalf + ceil(Int, AB / 2)

solveA = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, 0.1], A)
solveB = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-0.1, 1], B)

wrap_solver(solver) = u -> begin
    l = u[1]
    r = u[end]
    _, u = solver(l, r)
    #return [l; u; r] #maybe you need to pad back the bcs
    return u
end

f1 = wrap_solver(solveA)
f2 = wrap_solver(solveB)

p1 = zeros(AB, A)
p1[1:AB, end-AB+1:end] .= I(AB)
p1

p2 = zeros(AB, B)
p2[1:AB, 1:AB] = I(AB)
p2

s = CellularSheaf([A, B], [AB])
set_edge_maps!(s, 1, 2, 1, p1, p2)

hp = CollocationHP([f1, f2], s)


primal_sol, dual_sol = solve(hp, ADMM(2.0, 1))

u1 = primal_sol[Block(1)]
u2 = primal_sol[Block(2)]


u_solA = vcat(u1[1:end-AB], u2)
u_solB = vcat(u1, u2[AB+1:end])
@show norm(u_solA - u_solB)


p = scatter!(p, x, u_solA, label="hpA")
p = scatter!(p, x, u_solB, label="hpB")
# plot!(p, u, color=:teal, label="reference")
# plot!(p, rhs_func.(x), color=:purple, label="b")