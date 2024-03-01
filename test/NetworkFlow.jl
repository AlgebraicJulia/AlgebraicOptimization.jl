using AlgebraicOptimization.FinSetAlgebras
using AlgebraicOptimization.OpenFlowGraphs
using AlgebraicOptimization.Objectives
using AlgebraicOptimization.Optimizers
using Catlab
using Random
using StatsBase
using Test
using Plots
theme(:ggplot2)

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
    #return x -> x^2
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
    flows = random_flow(N, 4)
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
#=
g = random_open_flowgraph(10, .2, 3)
A = node_incidence_matrix(data(g))
p = to_problem(g)
o1 = Euler(gradient_flow(p), 0.01)
o2 = dual_decomposition(g, 0.01)

dual_sol1 = simulate(o1, zeros(10), 10)
dual_sol2 = simulate(o2, zeros(10), 10)

primal_sol1 = primal_solution(data(p), dual_sol1)
primal_sol2 = primal_solution(data(p), dual_sol2)

f(x) = sum([data(g).edge_costs[i](x[i]) for i in 1:nedges(data(g))]) + dual_sol1'*(A*x - data(g).flows)
=#



# Test flow graph composition
g1 = random_open_flowgraph(10, .2, 2)
g2 = random_open_flowgraph(10, .2, 3)
g3 = random_open_flowgraph(10, .2, 3)

g_comp = oapply(d, [g1, g2, g3])

# Test naturality of MCNF
γ = 0.1
iters = 20
p1 = to_problem(g1)
p2 = to_problem(g2)
p3 = to_problem(g3)
p_comp1 = oapply(d, [p1, p2, p3])

p_comp2 = to_problem(g_comp)

opt1 = Euler(gradient_flow(p_comp1), γ)
opt2 = Euler(gradient_flow(p_comp2), γ)

r11 = simulate(opt1, zeros(length(opt1.S)), iters)
r22 = simulate(opt2, zeros(length(opt2.S)), iters)

@test r11 ≈ r22

#o1 = dual_decomposition(g1, 0.1)
#o2 = dual_decomposition(g2, 0.1)
#o3 = dual_decomposition(g3, 0.1)

o1 = Euler(gradient_flow(p1),γ)
o2 = Euler(gradient_flow(p2),γ)
o3 = Euler(gradient_flow(p3),γ)

comp_opt1 = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])
comp_opt2 = dual_decomposition(g_comp, γ)

res1 = @time simulate(comp_opt1, zeros(length(comp_opt1.S)), iters)
res2 = @time simulate(comp_opt2, zeros(length(comp_opt2.S)), iters)

#@test res1 ≈ res2
#@test r11 ≈ res1

#opt3 = Euler(gradient_flow(p3), 0.1)

#r = simulate(opt3, zeros)


function graph_size_benchmark(d, node_sizes, connectivity, ss, iters)
    dd_times = []
    dd_mem = []
    hdd_times = []
    hdd_mem = []
    for N in node_sizes
        gs = [random_open_flowgraph(N, connectivity, length(ports(d, i))) for i in 1:nboxes(d)]
        g_comp = oapply(d, gs)
        dd_optimizer = dual_decomposition(g_comp, ss)
        os = [dual_decomposition(g, ss) for g in gs]
        hdd_optimizer = oapply(OpenDiscreteOpt(), d, os)
        λ0 = zeros(nvertices(data(g_comp)))
        res1 = simulate(dd_optimizer, λ0, iters)
        res2 = simulate(hdd_optimizer, λ0,iters)
        @test res1 ≈ res2
        t1 = @timed simulate(dd_optimizer, λ0, iters)
        t2 = @timed simulate(hdd_optimizer, λ0, iters)
        push!(dd_times, t1[2])
        push!(dd_mem, t1[3])
        push!(hdd_times, t2[2])
        push!(hdd_mem, t2[3])
        #push!(dd_times, @elapsed simulate(dd_optimizer, λ0, iters))
        #push!(hdd_times, @elapsed simulate(hdd_optimizer, λ0, iters))
    end
    return dd_times, hdd_times, dd_mem ./ (1024^3), hdd_mem ./ (1024^3)
end

function graph_connectivity_benchmark(d, num_nodes, connectivities, ss, iters)
    dd_times = []
    dd_mem = []
    hdd_times = []
    hdd_mem = []
    for p in connectivities
        gs = [random_open_flowgraph(num_nodes, p, length(ports(d, i))) for i in 1:nboxes(d)]
        g_comp = oapply(d, gs)
        dd_optimizer = dual_decomposition(g_comp, ss)
        os = [dual_decomposition(g, ss) for g in gs]
        hdd_optimizer = oapply(OpenDiscreteOpt(), d, os)
        λ0 = zeros(nvertices(data(g_comp)))
        res1 = simulate(dd_optimizer, λ0, iters)
        res2 = simulate(hdd_optimizer, λ0,iters)
        @test res1 ≈ res2

        t1 = @timed simulate(dd_optimizer, λ0, iters)
        t2 = @timed simulate(hdd_optimizer, λ0, iters)
        push!(dd_times, t1[2])
        push!(dd_mem, t1[3])
        push!(hdd_times, t2[2])
        push!(hdd_mem, t2[3])
        #push!(dd_times, @elapsed simulate(dd_optimizer, λ0, iters))
        #push!(hdd_times, @elapsed simulate(hdd_optimizer, λ0, iters))
    end
    return dd_times, hdd_times, dd_mem ./ (1024^3), hdd_mem ./ (1024^3)
end

f = "Computer Modern"
node_sizes = 10:10:120
#node_sizes = 10:5:30
connectivity = .2
ss = 0.01
iters = 10
dd_ts, hdd_ts, dd_mem, hdd_mem = graph_size_benchmark(d, node_sizes, connectivity, ss, iters)
p1 = plot(node_sizes, dd_ts, label="Standard DD",
    size = (1000,800),
    title="Performance vs. Graph Size",
    titlefont = (14,f),
    linewidth = 2,
    xlabel="Number of nodes per subgraph",
    ylabel ="Execution time (s)",
    thickness_scaling = 2,
    tickfont = (10,f),
    legend = :topleft,
    legend_font_family = f,
    #smooth = true,
    legendfontsize=10,
    seriescolor=palette(:default)[1],
    ms=4,
    guidefont=(f,12)
)
plot!(node_sizes, hdd_ts, label="Hierarchical DD", linewidth=2, seriescolor=palette(:default)[2])
p3 = plot(node_sizes, dd_mem, label="Standard DD",
    size = (1000,800),
    title="Memory Usage vs. Graph Size",
    titlefont = (14,f),
    linewidth = 2,
    xlabel="Number of nodes per subgraph",
    ylabel ="Memory used (GiB)",
    thickness_scaling = 2,
    tickfont = (10,f),
    legend = :topleft,
    legend_font_family = f,
    #smooth = true,
    legendfontsize=10,
    seriescolor=palette(:default)[1],
    ms=4,
    guidefont=(f,12)
)
plot!(node_sizes, hdd_mem, label="Hierarchical DD", linewidth=2, seriescolor=palette(:default)[2])





#num_nodes = 60
num_nodes = 70
connectivities = 0.1:0.1:1.0

dd_ts, hdd_ts, dd_mem, hdd_mem = graph_connectivity_benchmark(d, num_nodes, connectivities, ss, iters)
p2 = plot(connectivities, dd_ts, label="Standard DD",
    size = (1000,800),
    title="Performance vs. Graph Connectivity",
    titlefont = (14,f),
    linewidth = 2,
    xlabel="Connectivity factor per subgraph",
    ylabel ="Execution time (s)",
    thickness_scaling = 2,
    tickfont = (10,f),
    legend = :topleft,
    legend_font_family = f,
    #smooth = true,
    legendfontsize=10,
    seriescolor=palette(:default)[1],
    ms=4,
    guidefont=(f,12)
)
plot!(connectivities, hdd_ts, label="Hierarchical DD",linewidth=2, seriescolor=palette(:default)[2])

p4 = plot(connectivities, dd_mem, label="Standard DD",
    size = (1000,800),
    title="Memory Usage vs. Graph Connectivity",
    titlefont = (14,f),
    linewidth = 2,
    xlabel="Connectivity factor per subgraph",
    ylabel ="Memory used (GiB)",
    thickness_scaling = 2,
    tickfont = (10,f),
    legend = :topleft,
    legend_font_family = f,
    #smooth = true,
    legendfontsize=10,
    seriescolor=palette(:default)[1],
    ms=4,
    guidefont=(f,12)
)
plot!(connectivities, hdd_mem, label="Hierarchical DD",linewidth=2, seriescolor=palette(:default)[2])

##### Nice Benchmark Plots #####

l = @layout [a b; c d]
plot(p1,p2,p3,p4, layout=l, size=(2200,1600))


