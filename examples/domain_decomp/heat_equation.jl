using AlgebraicOptimization
using SparseArrays
using LinearAlgebra
using Plots
using DataInterpolations


# SETUP
#######

include("fnc_utils.jl")

N = 1500 # Number of nodes in the global problem.

# GLOBAL PROBLEM SETUP AND SOLVE
################################


bump(x, mu=0, sigma=10) = begin
    return exp.(-(x .- mu) .^ 2 ./ sigma)
end


uleft = -1
uright = +1
rhs_func(x) = 2bump(x, 1 / 2, 1 / 20) - 4bump(x, -1 / 4, 1 / 20)

# Construct 1st and 2nd differentiation matrices and a vector of nodes for the global problem.
x, Dₓ, Dₓₓ = diffmat2(N - 1, (-1, 1))

# Compute the correct global solution for comparison.
solveN = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [uleft, uright], N - 1)
x, u = solveN(-1, 1)
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
B = N - A + AB
@show length(x), N, A, B, A + B - AB, AB

# Create the local solvers for each subdomain.
solveA = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [-1, xAright], A - 1)
solveB = bvplin_solver(1 / 20, zero, zero, x -> -rhs_func(x), [xBleft, 1], B - 1)

# "wrap" a solver to just take in a u and return the projection of that u to the nearest solution.
wrap_solver(solver) = u -> begin
    l = u[1]
    r = u[end]
    x, u = solver(l, r)
    return x, u
end

# Create wrapped local solvers.
f1 = wrap_solver(solveA)
f2 = wrap_solver(solveB)



function alternating_projection(u₀, niter=1)
    xa, ua = f1(zeros(A + 1))
    xb, ub = f2(zeros(B + 1))
    function update(u1, u2)
        u2func = LinearInterpolation(u2, xb)
        u1[end] = u2func(xAright)
        # u2_res = p2 * u2
        # u1[end-AB+1:end] .= u2_res
        xa, u1 = f1(u1)
        u1func = LinearInterpolation(u1, xa)
        # u1_res = p1 * u1
        # u2[1:AB] .= u1_res
        u2[1] = u1func(xBleft)
        xb, u2 = f2(u2)
        return u1, u2
    end
    u1 = u₀[1:A]
    u2 = u₀[N-B+1:end]
    for i in 1:niter
        u1, u2 = update(u1, u2)
    end
    return (xa, u1), (xb, u2)
end

# Plot both the solutions and the error over iterations

begin
    rplt = plot(xlabel="x", ylabel="error", title="Error 1:$A, $(N-B):$N")
    rplt_tail = plot(xlabel="x", ylabel="error", title="Error 1:$A, $(N-B):$N")
    iters = [1, 5, 10, 25, 50, 100, 200]
    # iters = [50, 100, 200, 500]
    for i in iters
        # (xa, u1),(xb, u2) = alternating_projection(u, i)
        (xa, u1), (xb, u2) = alternating_projection(collect(LinRange(uleft, uright, N + 1)), i)
        ualt_func(x) = x < xAright ? LinearInterpolation(u1, xa)(x) : LinearInterpolation(u2, xb)(x)

        ualt = ualt_func.(x)
        plot!(p, xa, u1, label="ualt_$i(a)", linestyle=:dash, lw=2)
        plot!(p, xb, u2, label="ualt_$i(b)", linestyle=:dash, lw=2)
        if i < 100
            plot!(rplt, x, ualt - u, label="resid_$i", lw=2, ls=:dash)
        else
            scatter!(rplt_tail, x, log.(abs.((ualt - u) ./ (u .+ eps(Float64)))), label="resid_$i", lw=2, ls=:dash)
        end
        res = norm(ualt - u) / sqrt(N)
        relres = norm((ualt - u) ./ u) / sqrt(N)
        println("MSE of ualt_$i: $res\t Relative MSE $relres")
    end
    vline!(p, [xBleft, xAright], linestyle=:dash)
    vline!(rplt, [xBleft, xAright], linestyle=:dash)
    vline!(rplt_tail, [xBleft, xAright], linestyle=:dash)
    plt = plot(p, rplt, rplt_tail, layout=[1; 1; 1], size=(800, 700))
end
plt


# CELLULAR SHEAF SETUP
######################

# Make cellular sheaf.
#s = CellularSheaf([A, B], [AB])
#set_edge_maps!(s, 1, 2, 1, p1, p2)

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