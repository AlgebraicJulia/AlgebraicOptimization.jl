using AlgebraicOptimization
using Graphs
using SparseArrays
using LinearAlgebra
using Krylov
using Plots
using BlockArrays


# SETUP
#######

include("fnc_utils.jl")

N = 101 # Number of nodes in the global problem.

# GLOBAL PROBLEM SETUP AND SOLVE
################################

# Construct 1st and 2nd differentiation matrices and a vector of nodes for the global problem.
x, Dₓ, Dₓₓ = diffmat2(N - 1, (-1, 1))

bump(x, mu=0, sigma=10) = begin
    z = exp.(-(x .- mu) .^ 2 ./ sigma)
end


rhs_func(x) = 2bump(x, 1 / 2, 1 / 20) - 4bump(x, -1 / 2, 1 / 20)
# x, u = bvplin(1 / 20, x -> 0, x -> 0, x -> -rhs_func(x), [-1, 1], -1 / 2, -1 / 2, N)
# x, u = bvplin(1 / 20, x -> 0, x -> 0, x -> -rhs_func(x), [-1, 1], 0,0, N)

# Compute the correct global solution for comparison.
solveN = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, 1], N)
x, u = solveN(0, 0)

p = plot(x, rhs_func.(x), label="b")
p = plot!(p, x, u, label="bvplin_soln")

# LOCAL SOLVER SETUP AND SOLVE
##############################


# Index arithmetic to divide the mesh with AB amount of overlap.
Nhalf = ceil(Int, N / 2)
AB = 10
A = Nhalf + ceil(Int, AB / 2)
B = Nhalf + ceil(Int, AB / 2)


# Create the local solvers for each subdomain.
solveA = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, 0.1], A)
solveB = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-0.1, 1], B)

# "wrap" a solver to just take in a u and return the projection of that u to the nearest solution.
wrap_solver(solver) = u -> begin
    l = u[1]
    r = u[end]
    _, u = solver(l, r)
    #return [l; u; r] #maybe you need to pad back the bcs
    return u
end

# Create wrapped local solvers.
f1 = wrap_solver(solveA)
f2 = wrap_solver(solveB)


# CELLULAR SHEAF SETUP
######################


# make restriction maps. In this case, they are both projections.
p1 = zeros(AB, A)
p1[1:AB, end-AB+1:end] .= I(AB)
p1

p2 = zeros(AB, B)
p2[1:AB, 1:AB] .= I(AB)
p2

# Make cellular sheaf.
s = CellularSheaf([A, B], [AB])
set_edge_maps!(s, 1, 2, 1, p1, p2)

# Set up homological program using the local solvers we defined earlier.
hp = CollocationHP([f1, f2], s)

# Use ADMM to solve the HP.
primal_sol, dual_sol = solve(hp, ADMM(2.0, 1))

function lift_matching_family(primal_sol)
    u1 = primal_sol[Block(1)]
    u2 = primal_sol[Block(2)]
    u_solA = vcat(u1[1:end-AB], u2)
    u_solB = vcat(u1, u2[AB+1:end])
    return u_solA, u_solB
end

u_solA, u_solB = lift_matching_family(primal_sol)
@show norm(u_solA - u_solB)


# p = scatter!(p, x, u_solA, label="hpA")
# p = scatter!(p, x, u_solB, label="hpB")
# plot!(p, u, color=:teal, label="reference")
# plot!(p, rhs_func.(x), color=:purple, label="b")

u₀ = vcat(u[1:A], u[N-B+1:end])
primal_sol, dual_sol = solve(hp, ADMM(2.0, 1), u₀)
u_solA, u_solB = lift_matching_family(primal_sol)
@show norm(u_solA - u_solB)
p = scatter!(p, x, u_solB, label="hp-fp")

function alternating_projection(u₀, niter=1)
    function update(u1, u2)
        u1, u2 = f1(u1), f2(u2)
        mid = (p1 * u1 + p2 * u2) / 2
        @show length(mid)
        # y1 = vcat(u1[1:end-AB], u2)
        # y2 = vcat(u1, u2[AB+1:end])
        u1[end-AB+1:end] = mid
        u2[1:AB] = mid
        # avg_y = (y1+y2)/2
        @show length(u1)
        @show length(u2)
        # @show length(y1)
        # @show length(y2)
        # return avg_y[1:A-1], avg_y[N-B+1:end]
        return u1, u2
    end
    # u1 = u₀[Block(1)]
    # u2 = u₀[Block(2)]
    u1 = u₀[1:A]
    u2 = u₀[N-B:end]
    for i in 1:niter
        u1, u2 = update(u1, u2)
    end
    return vcat(u1, u2[AB+2:end])
end

u1 = f1(primal_sol[Block(1)])
u2 = f2(primal_sol[Block(2)])

ualt = alternating_projection(u₀, 2)
scatter!(p, x, ualt[1:end], label="ualt")

for i in [1, 2, 5, 10, 30, 100]
    ualt = alternating_projection(zeros(N), i)
    plot!(p, x, ualt, label="ualt_$i", marker=:none)
end
p

# @show norm(y1-y2)

# scatter!(p, x, y1, label="one_step_noproj A")
# scatter!(p, x, y2, label="one_step_noproj B")
# scatter!(p, x, avg_y, label="one_step_noproj mid")

# # plt2 = scatter(y1[Nhalf-AB:Nhalf+AB], label="y1")
# # plt2 = scatter!(plt2, y2[Nhalf-AB:Nhalf+AB], label="y2")

# x[N-B+1:A]
# p