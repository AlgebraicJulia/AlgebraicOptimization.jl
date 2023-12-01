""" Define the UWD algebra of open flow graphs
"""
module FlowGraphs

export FlowGraph, to_problem, node_incidence_matrix

using Catlab
import Catlab: src, tgt

using ..OpenProblems

struct FlowGraph
    exposed_nodes::Int
    nodes::Int
    edges::Int
    src::FinFunction # E -> V
    tgt::FinFunction # E -> V
    edge_costs::Vector{Function}
    flows::Vector{Float64}
    p::FinFunction
    FlowGraph(ens, ns, es, src, tgt, e_cs, fs, p) = 
        dom(p) != FinSet(ens) || codom(p) != FinSet(ns) ?
            error("Invalid portmap") : new(ens, ns, es, src, tgt, e_cs, fs, p)
end

# Construct a flow graph from a Catlab graph
function FlowGraph(g::Graph, costs, flows, portmap)
    V = nv(g)
    exposed_V = length(dom(portmap))
    return FlowGraph(exposed_V, V, ne(g), FinFunction(src(g)), FinFunction(tgt(g)), costs, flows, portmap)
end

# Convenience constructor when `nodes`==`exposed_nodes` and `p` is the identity function
FlowGraph(nodes, src, tgt, edge_costs, flows) =
    FlowGraph(nodes, nodes, src, tgt, edge_costs, flows, FinFunction(1:nodes))

nvertices(g::FlowGraph) = g.nodes
n_exposed_vertices(g::FlowGraph) = g.exposed_nodes
n_edges(g::FlowGraph) = g.edges
portmap(g::FlowGraph) = g.p
src(g::FlowGraph, e::Int) = g.src(e)
tgt(g::FlowGraph, e::Int) = g.tgt(e)

function node_incidence_matrix(g::FlowGraph)
    V = nvertices(g)
    E = n_edges(g)
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

function to_problem(g::FlowGraph)
    A = node_incidence_matrix(g)
    obj = (x,λ) -> sum([g.edge_costs[i](x[i]) for i in 1:n_edges(g)]) + λ'*(A*x - g.flows)

    return DualMaxProblem(
        n_exposed_vertices(g),
        nvertices(g),
        n_edges(g),
        obj,
        portmap(g)
    )
end

fills(d::AbstractUWD, b::Int, g::FlowGraph) = 
    b <= nparts(d, :Box) ? length(incident(d, b, :box)) == n_exposed_vertices(g) : 
        error("Trying to fill box $b when $d has fewer than $b boxes")

### oapply helper functions

induced_ports(d::AbstractUWD) = nparts(d, :OuterPort)
induced_ports(d::RelationDiagram) = subpart(d, [:outer_junction, :variable])

# Returns the pushout which induces the new set of vertices
function induced_vertices(d::AbstractUWD, gs::Vector{FlowGraph}, inclusions::Function)
    for b in parts(d, :Box)
        fills(d, b, gs[b]) || error("$(gs[b]) does not fill box $b")
    end

    total_portmap = copair([compose(portmap(gs[i]), inclusions(i)) for i in 1:length(gs)])

    #return pushout(FinFunction(subpart(d, :junction), nparts(d, :Junction)), total_portmap)
    return pushout(total_portmap, FinFunction(subpart(d, :junction), nparts(d, :Junction)))
end

function induced_graph(d::AbstractUWD, gs::Vector{FlowGraph}, var_map::FinFunction, inclusions::Function)
    #proj_mats = Matrix[]
    for b in parts(d, :Box)
        inc = compose(inclusions(b), var_map)
        push!(proj_mats, induced_matrix(inc))
    end
    return z::Vector -> sum([ps[b](proj_mats[b]*z) for b in 1:length(ps)])
end





end # module