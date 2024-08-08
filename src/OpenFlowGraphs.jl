module OpenFlowGraphs

export FlowGraph, underlying_graph, FG, OpenFG, to_problem,
    node_incidence_matrix, dual_decomposition, nvertices, nedges, random_open_flowgraph

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using ..Objectives
using ..Optimizers
using Catlab
import Catlab: oapply, dom, src, tgt
using Test
using Optim
using ForwardDiff
using StatsBase
using Random

struct FlowGraph
    nodes::FinSet
    edges::FinSet
    src::FinFunction
    tgt::FinFunction
    edge_costs::Vector{Function}
    flows::Vector{Float64}
end

function FlowGraph(g::Graph, costs, flows)
    return FlowGraph(FinSet(nv(g)), FinSet(ne(g)), 
        FinFunction(src(g), nv(g)), 
        FinFunction(tgt(g), nv(g)), 
        costs, flows)
end

src(g::FlowGraph, e::Int) = g.src(e)
tgt(g::FlowGraph, e::Int) = g.tgt(e)

dom(g::FlowGraph) = FinSet(g.nodes)
function underlying_graph(g::FlowGraph)
    res = Graph(length(g.nodes))
    add_edges!(res, g.src.func, g.tgt.func)
    return res
end

nvertices(g::FlowGraph) = length(g.nodes)
nedges(g::FlowGraph) = length(g.edges)

struct FG <: FinSetAlgebra{FlowGraph} end

hom_map(::FG, ϕ::FinFunction, g::FlowGraph) =
    FlowGraph(codom(ϕ), g.edges, 
        g.src⋅ϕ, g.tgt⋅ϕ, 
        g.edge_costs, pushforward_function(ϕ, g.flows))

function laxator(::FG, gs::Vector{FlowGraph})
    laxed_src = reduce(⊕, [g.src for g in gs])
    laxed_tgt = reduce(⊕, [g.tgt for g in gs])
    laxed_flows = vcat([g.flows for g in gs]...)
    laxed_costs = vcat([g.edge_costs for g in gs]...)
    return FlowGraph(codom(laxed_src), dom(laxed_src),
        laxed_src, laxed_tgt, laxed_costs, laxed_flows)
end

struct OpenFG <: CospanAlgebra{Open{FlowGraph}} end

function oapply(d::AbstractUWD, gs::Vector{Open{FlowGraph}})
    return oapply(OpenFG(), FG(), d, gs)
end

# Flow graphs to min cost net flow objective functions
function node_incidence_matrix(g::FlowGraph)
    V = nvertices(g)
    E = nedges(g)
    A = zeros(V,E)
    for (v,e) in Iterators.product(1:V, 1:E)
        if src(g, e) == tgt(g, e) && tgt(g, e) == v
            continue
        elseif src(g,e) == v
            A[v,e] = -1
        elseif tgt(g,e) == v
            A[v,e] = 1
        end
    end
    return A
end

function to_problem(og::Open{FlowGraph})
    g = data(og)
    S = og.S
    m = og.m
    A = node_incidence_matrix(g)
    function obj(x,λ)
        return sum([g.edge_costs[i](x[i]) for i in 1:nedges(g)]) + λ'*(A*x-g.flows)
    end

    return Open{SaddleObjective}(S, SaddleObjective(FinSet(nedges(g)), S, obj), m)
end

function dual_decomposition(og::Open{FlowGraph}, γ)
    g = data(og)
    A = node_incidence_matrix(g)
    N = nedges(g)
    
    L(i) = (x,λ) -> g.edge_costs[i](x[1]) + λ'*(A[:,i]*x[1] - 1/N*g.flows)
    L(x,λ) = sum([L(i)(x[i], λ) for i in 1:N])

    function dual_decomp_dynamics(λ)
        x = zeros(N)
        #=Threads.@threads=# for i in 1:N
            x[i] = (optimize(x -> L(i)(x,λ), [0.0], LBFGS(), autodiff=:forward).minimizer)[1]
        end
        return λ + γ*ForwardDiff.gradient(λ->L(x,λ), λ)
    end
    return Open{Optimizer}(og.S, dual_decomp_dynamics, og.m)
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



end