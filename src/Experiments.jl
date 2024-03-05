using NLsolve
using LinearAlgebra
using ForwardDiff
using Catlab

function node_incidence_matrix(g::Graph)
    V = nv(g)
    E = ne(g)
    A = zeros(V,E)
    for (v,e) in Iterators.product(1:V, 1:E)
        if src(g, e) == tgt(g, e) && tgt(g, e) == v
            continue
        elseif src(g,e) == v
            A[v,e] = 1
        elseif tgt(g,e) == v
            A[v,e] = -1
        end
    end
    return A
end

N_subprob = 10
A1 = node_incidence_matrix(wheel_graph(Graph, N_subprob))
A2 = A1
A3 = A1







function draw2(n::Int)
    a = rand(1:n)
    b = rand(1:n-1)
    b += (b >= a)
    return a, b
end

function random_column(length::Int)
    a,b = draw2(length)
    res = zeros(length)
    res[a] = 1
    res[b] = -1
    return res
end

E = 50
V = 30
num_sources = 5
num_sinks = 5

A = hcat([random_column(V) for i in 1:E]...)

f(x) = x^2

b = zeros(V)

for i in 1:num_sources
    src_vertex = rand(1:V)
    val = rand(1:0.01:10)
    b[src_vertex] = val
end

for i in 1:num_sinks
    sink_vertex = rand(1:V)
    val = rand(1:0.01:10)
    b[sink_vertex] = -val
end
λ = randn(V)
total_L(x) = sum([f(x_i) for x_i in x]) + λ'*(A*x-b) 

L(i) = x -> f(x) + λ'*(A[:,i]*x - b)

function grad_flow_L(x::Vector)
    ForwardDiff.gradient(total_L, x)
end

function grad_flow_L(i::Int)
    x -> ForwardDiff.derivative(L(i), x)
end

function grad_descent(f, x0, γ, max_iters, ϵ)
    x_prev = x0
    x_cur = x0
    for i in 1:max_iters
        x_cur = x_prev - γ*ForwardDiff.gradient(f, x_prev)
        if norm(f(x_cur) - f(x_prev)) < ϵ
            #println("Terminated in $i iterations.")
            return x_cur
        end
        x_prev = x_cur
    end
    println("Did not converge.")
    return x_cur
end

function grad_descent_1D(f, x0, γ, max_iters, ϵ)
    x_prev = x0
    x_cur = x0
    for i in 1:max_iters
        x_cur = x_prev - γ*ForwardDiff.derivative(f, x_prev)
        if norm(f(x_cur) - f(x_prev)) < ϵ
            #println("Terminated in $i iterations.")
            return x_cur
        end
        x_prev = x_cur
    end
    println("Did not converge.")
    return x_cur
end

sol_nl_total = nlsolve(grad_flow_L, repeat([10.0], E), iterations=1000000, xtol=0.01).zero
@time nlsolve(grad_flow_L, repeat([10.0], E), iterations=1000000, xtol=0.01)

function sol_nl(i::Int) 
    f = grad_flow_L(i)
    sol = nlsolve(n_ary(f), [10.0], xtol=0.01)
    return sol.zero[1]
end

sol_nl_distributed = zeros(E)
for i in 1:E
    sol_nl_distributed[i]=sol_nl(i)
end

@time for i in 1:E
    sol_nl(i)
end

sol_total = grad_descent(total_L, repeat([10.0], E), 0.1,100000, 0.0001)
@time grad_descent(total_L, repeat([10.0], E), 0.1,100000, 0.01)
sol(i) = grad_descent_1D(L(i), 10.0, 0.1, 100000, 0.01)

sol_distributed = zeros(E)

for i in 1:E
    sol_distributed[i] = sol(i)
end

@time for i in 1:E
    sol(i)
end

