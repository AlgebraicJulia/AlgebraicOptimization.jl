module OpenFlowGraphs

export FlowGraph, underlying_graph, FG, OpenFG, to_problem,
    node_incidence_matrix

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using ..Objectives
using ..Optimizers
using Catlab
import Catlab: oapply, dom, src, tgt
using Test

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
        g.edge_costs, pushforward_matrix(ϕ)*g.flows)

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
            A[v,e] = 1
        elseif tgt(g,e) == v
            A[v,e] = -1
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
        sum([g.edge_costs[i](x[i]) for i in 1:nedges(g)]) 
        + λ'*(A*x-g.flows)
    end

    return Open{SaddleObjective}(S, SaddleObjective(FinSet(nedges(g)), S, obj), m)
end

end