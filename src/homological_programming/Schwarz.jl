using LinearAlgebra
using Optim
using Plots

subproblem_dim = 10
global_dim = subproblem_dim * 2 - 2

a = rand(subproblem_dim * 2 - 2)

Q = rand(subproblem_dim, subproblem_dim)
Q = Q' * Q
c = rand(subproblem_dim)

R = rand(subproblem_dim, subproblem_dim)
R = R' * R
d = rand(subproblem_dim)

A = rand(global_dim, global_dim)
A = A' * A

f(x) = x' * A[1:subproblem_dim, 1:subproblem_dim] * x - a[1:subproblem_dim]' * x
g(y) = y' * A[subproblem_dim-1:end, subproblem_dim-1:end] * y - a[subproblem_dim-1:end]' * y


subproblem1(x_end) = x -> f(vcat(x, x_end))
subproblem2(y_start) = y -> g(vcat(y_start, y))

total_problem(z) = f(z[1:subproblem_dim]) + g(z[subproblem_dim-1:end])
total_problem2(z) = z' * A * z - a' * z

function alternate(niters, x0, y0)
    x_res = [x0]
    y_res = [y0]
    for i in 1:niters
        x_cur = x_res[end]
        y_cur = y_res[end]
        x_update = vcat(Optim.minimizer(optimize(subproblem1(y_cur[2]), x_cur[1:end-1], LBFGS())), y_cur[2])
        y_update = vcat(x_update[end-1], Optim.minimizer(optimize(subproblem2(x_update[end-1]), y_cur[2:end], LBFGS())))

        push!(x_res, x_update)
        push!(y_res, y_update)
    end

    return x_res, y_res
end

true_solution = Optim.minimizer(optimize(total_problem, rand(subproblem_dim * 2 - 2), LBFGS()))
true_solution2 = Optim.minimizer(optimize(total_problem2, rand(subproblem_dim * 2 - 2), LBFGS()))

x_res, y_res = alternate(10, rand(subproblem_dim), rand(subproblem_dim))

x_sol = x_res[end]
y_sol = y_res[end]

println(norm(x_sol[end-1:end] - y_sol[1:2]))

our_solution = vcat(x_sol, y_sol[3:end])

computed_trajectory = [vcat(x, y[3:end]) for (x, y) in zip(x_res, y_res)]

losses = total_problem.(computed_trajectory)

println(norm(our_solution - true_solution))
println(norm(our_solution - true_solution2))

plot(losses)


