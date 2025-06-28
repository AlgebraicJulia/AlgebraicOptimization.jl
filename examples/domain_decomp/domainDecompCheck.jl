using AlgebraicOptimization
using Graphs
using SparseArrays
using LinearAlgebra
using Krylov
using Plots
using BlockArrays
using Printf

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

# Compute the correct global solution for comparison.
solveN = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, 1], N-1)
x, u = solveN(-1, 1)
@show length(x), length(u)
p = plot(x, rhs_func.(x), label="b", lw=3, title="Solution n=$N")
p = plot!(p, x, u, label="bvplin_soln", lw=3, xlabel="x", ylabel="u", legend=:bottomright)

# LOCAL SOLVER SETUP AND SOLVE
##############################

# Index arithmetic to divide the mesh with AB amount of overlap.
Nhalf = ceil(Int, N / 2)

xAright = 0.1
xBleft = -0.1
AB = length(findall(xBleft .<= x .<= xAright))
A = Nhalf + ceil(Int, AB / 2)
# B = Nhalf + ceil(Int, AB / 2)-1
B = N-A+AB
@show length(x), N, A, B, A+B-AB, AB

# Create the local solvers for each subdomain.
solveA = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, xAright], A-1)
solveB = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [xBleft, 1], B-1)
# solveAB = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [xBleft, xAright], AB)

# "wrap" a solver to just take in a u and return the projection of that u to the nearest solution.
wrap_solver(solver) = u -> begin
    l = u[1]
    r = u[end]
    x, u = solver(l, r)
    # @show length(x), length(u)
    # @show minimum(x), maximum(x)
    #return [l; u; r] #maybe you need to pad back the bcs
    return u
end

# Create wrapped local solvers.
f1 = wrap_solver(solveA)
f2 = wrap_solver(solveB)
# f12 = wrap_solver(solveAB)

#--------------------------
# local solutions
u1 = f1(u[1:A])                 
u2 = f2(u[N-B+1:end])           

# corresponding meshes
xA, _, _ = diffmat2(A - 1, (x[1], xAright))       
xB, _, _ = diffmat2(B - 1, (xBleft, x[end]))        

# Extract last 20 of A and first 20 of B
n = ceil(Int, N/10)
println("n = $n")
xA_10 = xA[end-n+1:end]
u1_10 = u1[end-n+1:end]

xB_10 = xB[1:n]
u2_10 = u2[1:n]

println("\n|   xA    |   u1(x)   ||   xB   |   u2(x)  |")
println("--------------------------------------------------")
for i in 1:n
    xa = @sprintf("%7.4f", xA_10[i])
    ua = @sprintf("%8.5f", u1_10[i])
    xb = @sprintf("%7.4f", xB_10[i])
    ub = @sprintf("%8.5f", u2_10[i])
    println("| $xa | $ua || $xb | $ub |")
end