using AlgebraicOptimization.OpenProblems

using Catlab
using AlgebraicDynamics.UWDDynam
using Test
using ForwardDiff

d = @relation (x,y,z) begin
    f(x,w)
    g(y,w)
    h(z,w)
end

A = [1 2; 3 4]

f(x) = x[1]^2 + x[2]
g(y) = y'*A*y - [2,1]'*y
h(z) = z[1]^2 + z[2]^2


open_f = OpenProblem(2,f)
open_g = OpenProblem(2,g)
open_h = OpenProblem(2,h)

cp = oapply(d, [open_f,open_g,open_h])
true_cp(u) = f([u[1],u[2]]) + g([u[3],u[2]]) + h([u[4],u[2]])

test_u = rand(4)*10
#test_u = Float64[3,2,1,2]
@test cp(test_u) == true_cp(test_u)

@time cp(test_u)
@time true_cp(test_u)

@test ForwardDiff.gradient(objective(cp), test_u) == ForwardDiff.gradient(true_cp, test_u)

@time ForwardDiff.gradient(objective(cp), test_u)
@time ForwardDiff.gradient(true_cp, test_u)

# Test naturality of gradient flow

function iterate(f, x0, num_iters)
    x = x0
    for i in 1:num_iters
        x = f(x)
    end
    return x
end


γ = 0.01
x0 = repeat([100], 4)
num_iters=100

gf1 = oapply(d, gradient_flow([open_f, open_g, open_h]))
gd1 = euler_approx(gf1, γ)
#sol1 = trajectory(gd1, test_u, nothing, tspan)
sol1 = iterate(x -> eval_dynamics(gd1, x), x0, num_iters)

gf2 = gradient_flow(cp)
gd2 = euler_approx(gf2, γ)
#sol2 = trajectory(gd2, test_u, nothing, tspan)
sol2 = iterate(x -> eval_dynamics(gd2, x), x0, num_iters)

@test sol1 == sol2
