using AlgebraicDynamics.UWDDynam
using AlgebraicOptimization.OpenProblems
using AlgebraicOptimization.FlowGraphs
using Catlab
using NLsolve
using Optim

# Make a flow graph from a catlab graph

g = wheel_graph(Graph, 6)

flow_costs = [x -> #=rand(1:.1:10)=#e*x^2 for e in 1:ne(g)]
flows = zeros(nv(g))
flows[1] = 10
flows[nv(g)] = -10
pm = FinFunction([1,nv(g)])

fg = FlowGraph(g, flow_costs, flows, pm)
A = node_incidence_matrix(fg)

p = to_problem(fg)

s = gradient_flow(p)

#flow_s = λ -> eval_dynamics(s, λ)

ds = euler_approx(s, 0.1)


function iterate(f, x0, num_iters)
    x = x0
    for i in 1:num_iters
        x = f(x)
    end
    return x
end

dual_sol = iterate(u -> eval_dynamics(ds, u), zeros(nv(g)), 30)

#dual_sol = nlsolve(flow_s, zeros(nv(g)), xtol=0.01)

#primal_sol = primal_solution(p, dual_sol)
primal_sol = iterate(x -> x - 0.001*ForwardDiff.gradient(x->objective(p)(x,dual_sol), x), zeros(ne(g)), 100000)

function uzawas(L::Function, init_x, init_λ, γ_init, γ_decay, iters)
    x_old = init_x
    x_new = x_old
    y_old = init_λ
    y_new = y_old
    γ = γ_init
    for i in 1:iters
        x_new = x_old - γ*ForwardDiff.gradient(x->L(x,y_old), x_old)
        y_new = y_old + γ*ForwardDiff.gradient(y->L(x_old,y), y_old)
        x_old = x_new
        y_old = y_new
        γ -= γ_decay
    end
    return x_new, y_new
end

function dual_ascent(L::Function, primal_dim, init_y, γ, iters)
    y = init_y
    x(y) = optimize(x->L(x,y), zeros(primal_dim), LBFGS()#=, autodiff=:forward=#)
    for i in 1:iters
        x_star = x(y).minimizer
        y = y + γ*ForwardDiff.gradient(y->L(x_star,y), y)
    end
    return x(y).minimizer, y
end



ups, uds = uzawas(objective(p), zeros(ne(g)), zeros(nv(g)), .001, 0, 100000)
daps, dads = dual_ascent(objective(p), ne(g), zeros(nv(g)), 0.1, 100)


