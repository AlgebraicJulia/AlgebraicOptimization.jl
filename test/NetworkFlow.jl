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

d = @relation (a,y,z) begin
    f(w,x,a)
    g(u,w,y)
    h(u,w,x,z)
end





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
#node_sizes = 10:10:150
node_sizes = 10:5:30
connectivity = .2
ss = 0.01
iters = 10
dd_ts, hdd_ts, dd_mem, hdd_mem = graph_size_benchmark(d, node_sizes, connectivity, ss, iters)
p1 = plot(node_sizes, dd_ts, label="Standard DD",
    size = (1000,800),
    marker=:circle,
    title="Speed vs. Graph Size",
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
plot!(node_sizes, hdd_ts, label="Hierarchical DD", marker=:square, linewidth=2, seriescolor=palette(:default)[2])
p3 = plot(node_sizes, dd_mem, label="Standard DD",
    size = (1000,800),
    marker=:circle,
    title="Memory Usage vs. Graph Size",
    titlefont = (14,f),
    linewidth = 2,
    #xlabel="Number of nodes per subgraph",
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
plot!(node_sizes, hdd_mem, label="Hierarchical DD",marker=:square, linewidth=2, seriescolor=palette(:default)[2])

size_speedups = [x / y for (x,y) in zip(dd_ts, hdd_ts)]
size_rel_mem = [x / y for (x,y) in zip(dd_mem, hdd_mem)]

p5 = plot(node_sizes, size_speedups, label="Relative Speedup",
    size = (1000,800),
    marker=:circle,
    title="Relative Performance vs. Graph Size",
    titlefont = (14,f),
    linewidth = 2,
    #xlabel="Number of nodes per subgraph",
    #ylabel ="Execution time (s)",
    thickness_scaling = 2,
    tickfont = (10,f),
    legend = :topleft,
    legend_font_family = f,
    #smooth = true,
    legendfontsize=10,
    seriescolor=palette(:default)[3],
    ms=4,
    guidefont=(f,12)
)
plot!(node_sizes, size_rel_mem, label="Relative Memory Usage", marker=:square, linewidth=2, seriescolor=palette(:default)[4])



#num_nodes = 80
num_nodes = 10
connectivities = 0.1:0.1:1.0

dd_ts, hdd_ts, dd_mem, hdd_mem = graph_connectivity_benchmark(d, num_nodes, connectivities, ss, iters)
p2 = plot(connectivities, dd_ts, label="Standard DD",
    size = (1000,800),
    marker=:circle,
    title="Speed vs. Graph Connectivity",
    titlefont = (14,f),
    linewidth = 2,
    xlabel="Connectivity factor per subgraph",
    #ylabel ="Execution time (s)",
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
plot!(connectivities, hdd_ts, label="Hierarchical DD",marker=:square,linewidth=2, seriescolor=palette(:default)[2])

p4 = plot(connectivities, dd_mem, label="Standard DD",
    size = (1000,800),
    title="Memory Usage vs. Graph Connectivity",
    titlefont = (14,f),
    marker=:circle,
    linewidth = 2,
    #xlabel="Connectivity factor per subgraph",
    #ylabel ="Memory used (GiB)",
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
plot!(connectivities, hdd_mem, label="Hierarchical DD",marker=:square,linewidth=2, seriescolor=palette(:default)[2])

conn_speedups = [x / y for (x,y) in zip(dd_ts, hdd_ts)]
conn_rel_mem = [x / y for (x,y) in zip(dd_mem, hdd_mem)]
p6 = plot(connectivities, conn_speedups, label="Relative Speedup",
    size = (1000,800),
    title="Relative Performance vs. Graph Connectivity",
    titlefont = (14,f),
    marker=:circle,
    linewidth = 2,
    #xlabel="Connectivity factor per subgraph",
    #ylabel ="Memory used (GiB)",
    thickness_scaling = 2,
    tickfont = (10,f),
    legend = :topleft,
    legend_font_family = f,
    #smooth = true,
    legendfontsize=10,
    seriescolor=palette(:default)[3],
    ms=4,
    guidefont=(f,12)
)
plot!(connectivities, conn_rel_mem, label="Relative Memory Usage",marker=:square,linewidth=2, seriescolor=palette(:default)[4])

##### Nice Benchmark Plots #####

l = @layout [a b; c d; e f]
plot(p1,p2,p3,p4,p5,p6, layout=l, size=(2300,2400))


