using AlgebraicOptimization.FinSetAlgebras
using AlgebraicOptimization.OpenFlowGraphs
using AlgebraicOptimization.Objectives
using AlgebraicOptimization.Optimizers
using Catlab
using Random
using StatsBase

d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end

function random_connected_graph(nv, p)
    g = erdos_renyi(Graph, nv, p)
    while(length(connected_components(g))>1)
        g = erdos_renyi(Graph, nv, p)
    end
    return g
end

g = random_connected_graph(10, .2)
fg = FlowGraph(g, [], [])
@test underlying_graph(fg) == g

function random_quadratic()
    a = rand()
    b = rand()*rand([-1,1])
    c = rand()*rand([-1,1])
    return x -> a*x^2 + b*x + c
end

function random_flow(n::Int, n_nonzeros::Int)
    u = 2*rand(n_nonzeros) .- 1
    x = [(u[1]-u[n_nonzeros])/2; diff(u) ./ 2]
    res = vcat(x, zeros(n - n_nonzeros))
    return shuffle(res)
end

function random_flow_graph(N::Int, connectivity)
    g = random_connected_graph(N, connectivity)
    E = ne(g)
    flow_costs = [random_quadratic() for i in 1:E]
    flows = random_flow(N, 2)
    return FlowGraph(g, flow_costs, flows)
end

function random_injection(dom::Int, codom::Int)
    f = sample(1:codom, dom, replace=false)
    return FinFunction(sort(f), codom)
end

function random_open_flowgraph(n_vertices, p, n_boundary)
    return Open{FlowGraph}(FinSet(n_vertices), 
        random_flow_graph(n_vertices, p), 
        random_injection(n_boundary, n_vertices))
end

# Test flow graph composition
g1 = random_open_flowgraph(12, .2, 2)
g2 = random_open_flowgraph(10, .2, 3)
g3 = random_open_flowgraph(7, .2, 3)

g_comp = oapply(d, [g1, g2, g3])

# Test naturality of MCNF
p1 = to_problem(g1)
p2 = to_problem(g2)
p3 = to_problem(g3)
p_comp1 = oapply(d, [p1, p2, p3])

p_comp2 = to_problem(g_comp)

opt1 = Euler(gradient_flow(p_comp1), 0.1)
opt2 = Euler(gradient_flow(p_comp2), 0.1)

r1 = simulate(opt1, zeros(length(opt1.S)), 10)
r2 = simulate(opt2, zeros(length(opt2.S)), 10)

@test r1 == r2

#opt3 = Euler(gradient_flow(p3), 0.1)

#r = simulate(opt3, zeros)


