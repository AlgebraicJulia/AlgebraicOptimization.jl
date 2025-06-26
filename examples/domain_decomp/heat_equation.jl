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

N = 100 # Number of nodes in the global problem.

# GLOBAL PROBLEM SETUP AND SOLVE
################################

# Construct 1st and 2nd differentiation matrices and a vector of nodes for the global problem.
x, Dₓ, Dₓₓ = diffmat2(N - 1, (-1, 1))

bump(x, mu=0, sigma=10) = begin
    z = exp.(-(x .- mu) .^ 2 ./ sigma)
end


rhs_func(x) = 2bump(x, 1 / 2, 1 / 20) - 4bump(x, -2 / 3, 1 / 30)
# x, u = bvplin(1 / 20, x -> 0, x -> 0, x -> -rhs_func(x), [-1, 1], -1 / 2, -1 / 2, N)
# x, u = bvplin(1 / 20, x -> 0, x -> 0, x -> -rhs_func(x), [-1, 1], 0,0, N)

# Compute the correct global solution for comparison.
solveN = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, 1], N)
x, u = solveN(0, 0)

p = plot(x, rhs_func.(x), label="b", lw=3, title="Solution n=$N")
p = plot!(p, x, u, label="bvplin_soln", lw=3, xlabel="x", ylabel="u", legend=:bottomright)
plt2 = deepcopy(p)


# LOCAL SOLVER SETUP AND SOLVE
##############################

# Index arithmetic to divide the mesh with AB amount of overlap.
Nhalf = ceil(Int, N / 2)

xAright = 0.1
xBleft = -0.1
AB = length(findall(xBleft .<= x .<= xAright))
A = Nhalf + ceil(Int, AB / 2)
B = Nhalf + ceil(Int, AB / 2)


# Create the local solvers for each subdomain.
solveA = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, xAright], A)
solveB = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [xBleft, 1], B)

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


"""
Make restriction maps. In this case, they are both projections.
"""
function restriction_matrices(A, B, AB)
    p1 = zeros(AB, A)
    p1[1:AB, end-AB+1:end] .= I(AB)

    p2 = zeros(AB, B)
    p2[1:AB, 1:AB] .= I(AB)
    return p1, p2
end

p1, p2 = restriction_matrices(A, B, AB)

function alternating_projection(u₀, niter=1)
    function update(u1, u2)
        u1, u2 = f1(u1), f2(u2)
        mid = (p1 * u1 + p2 * u2) / 2
        u1[end-AB+1:end] = mid
        u2[1:AB] = mid
        return u1, u2
    end
    u1 = u₀[1:A]
    u2 = u₀[N-B:end]
    for i in 1:niter
        u1, u2 = update(u1, u2)
    end
    return vcat(u1, u2[AB+1:end])
end

# Plot both the solutions and the error over iterations

begin
    rplt = plot(xlabel="x", ylabel="error", title="Error 1:$A, $(N-B):$N")
    iters = [10, 50, 100, 200]
    for i in iters
        ualt = alternating_projection(zeros(N), i)
        plot!(p, x, ualt, label="ualt_$i", linestyle=:dash, lw=2)
        plot!(rplt, x, ualt - u, label="resid_$i", lw=2, ls=:dash)
        println("Residual of ualt_$i: ", norm(ualt - u))
    end
    vline!(p, [xBleft, xAright], linestyle=:dash)
    vline!(rplt, [xBleft, xAright], linestyle=:dash)
    plt = plot(p, rplt, layout=[1; 1], size=(800, 700))
end
plt


# CELLULAR SHEAF SETUP
######################

# Make cellular sheaf.
s = CellularSheaf([A, B], [AB])
set_edge_maps!(s, 1, 2, 1, p1, p2)

# Set up homological program using the local solvers we defined earlier.
# hp = CollocationHP([f1, f2], s)
# CELLULAR SHEAF SETUP
######################

# Use ADMM to solve the HP.
#primal_sol, dual_sol = solve(hp, ADMM(2.0, 1))

#=function lift_matching_family(primal_sol)
    u1 = primal_sol[Block(1)]
    u2 = primal_sol[Block(2)]
    u_solA = vcat(u1[1:end-AB], u2)
    u_solB = vcat(u1, u2[AB+1:end])
    return u_solA, u_solB
end =#

#u_solA, u_solB = lift_matching_family(primal_sol)
#@show norm(u_solA - u_solB)


# p = scatter!(p, x, u_solA, label="hpA")
# p = scatter!(p, x, u_solB, label="hpB")
# plot!(p, u, color=:teal, label="reference")
# plot!(p, rhs_func.(x), color=:purple, label="b")

#=u₀ = vcat(u[1:A], u[N-B+1:end])
primal_sol, dual_sol = solve(hp, ADMM(2.0, 1), u₀)
u_solA, u_solB = lift_matching_family(primal_sol)
@show norm(u_solA - u_solB)
p = scatter!(p, x, u_solB, label="hp-fp")=#


#u1 = f1(primal_sol[Block(1)])
#u2 = f2(primal_sol[Block(2)])