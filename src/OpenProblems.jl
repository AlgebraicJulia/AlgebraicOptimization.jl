""" Define the UWD algebra of open optimization problems
"""
module OpenProblems

export MinimizationProblem, DualMaxProblem, nvars, n_exposed_vars, portmap, objective, gradient_flow,
    primal_solution, primal_objective, induced_vars, induced_matrix, n_primal_vars

using Catlab
using AlgebraicDynamics.UWDDynam
using ForwardDiff
using NLsolve
using Optim
import Catlab.oapply

# Closed Problems
#################

struct DualFunction
    primal_part::Function
    dual_part::Function
end


# Open Problems
###############

abstract type AbstractProblem end

""" OpenProblem

An open optimization problem R^`vars` -> R. The problem exposes `exposed_vars` variables
given by the portmap `p`. The function `objective` should take `Vector{Float64}` of length
`vars` as input and return a scalar.
"""
struct MinimizationProblem <: AbstractProblem
    exposed_vars::Int
    vars::Int
    objective::Function
    p::FinFunction
    MinimizationProblem(evs, vs, obj, p) = dom(p) != FinSet(evs) || codom(p) != FinSet(vs) ?
        error("Invalid portmap") : new(evs, vs, obj, p)
end
# Convenience constructor when `vars`==`exposed_vars` and `p` is the identity function
MinimizationProblem(nvars, obj) = MinimizationProblem(nvars,nvars,obj,FinFunction(1:nvars))

# Make `OpenProblem`s callable
(p::MinimizationProblem)(z::Vector) = p.objective(z)

struct DualMaxProblem <: AbstractProblem
    exposed_vars::Int
    vars::Int
    primal_vars::Int
    objective::Function # x × λ → R
    #dual_objective::Function
    p::FinFunction
    DualMaxProblem(evs, vs, pvs, obj, p) = dom(p) != FinSet(evs) || codom(p) != FinSet(vs) ?
        error("Invalid portmap") : new(evs, vs, pvs, obj, p)
end
# Convenience constructor when `vars`==`exposed_vars` and `p` is the identity function
DualMaxProblem(nvars, pnvars, obj) = DualMaxProblem(nvars,nvars,pnvars,obj,FinFunction(1:nvars))

nvars(p::AbstractProblem) = p.vars
n_exposed_vars(p::AbstractProblem) = p.exposed_vars
objective(p::AbstractProblem) = p.objective
primal_objective(p::DualMaxProblem, λ) = x -> objective(p)(x,λ)
dual_objective(p::DualMaxProblem, x) = λ ->  objective(p)(x,λ)
portmap(p::AbstractProblem) = p.p

n_primal_vars(p::DualMaxProblem) = p.primal_vars

# Open problems as a UWD algebra
################################

""" fills(d::AbstractUWD, b::Int, p::OpenProblem)

Checks if `p` is of the correct signature to fill box `b` of the uwd `d`
"""
fills(d::AbstractUWD, b::Int, p::AbstractProblem) = 
    b <= nparts(d, :Box) ? length(incident(d, b, :box)) == n_exposed_vars(p) : 
        error("Trying to fill box $b when $d has fewer than $b boxes")

### oapply helper functions

induced_ports(d::AbstractUWD) = nparts(d, :OuterPort)
induced_ports(d::RelationDiagram) = subpart(d, [:outer_junction, :variable])

# Returns the pushout which induces the new set of variables
function induced_vars(d::AbstractUWD, ps::Vector, inclusions::Function)
    for b in parts(d, :Box)
        fills(d, b, ps[b]) || error("$(ps[b]) does not fill box $b")
    end

    total_portmap = copair([compose(portmap(ps[i]), inclusions(i)) for i in 1:length(ps)])

    #return pushout(FinFunction(subpart(d, :junction), nparts(d, :Junction)), total_portmap)
    return pushout(total_portmap, FinFunction(subpart(d, :junction), nparts(d, :Junction)))
end

# Takes a FinFunction from N->M and returns the induced linear map R^M->R^N
function induced_matrix(dom::Int, codom::Int, f#=::Vector{Int}=#)::Matrix{Float64}
    length(f) == dom && max(f...) <= codom || error("Invalid FinFunction.")
    res = zeros(dom, codom)
    for (i,j) in Iterators.product(1:dom, 1:codom)
        if f[i] == j
            res[i,j] = 1
        end
    end
    return res
end

induced_matrix(f::FinFunction) = induced_matrix(length(dom(f)), length(codom(f)), f.func)

# Sums objective functions of `ps` subject to correct variable sharing according to `d`.
# `var_map` should be the left leg of the induced variable pushout.
# `inclusions` should be a function mapping box numbers to the inclusion of that box's variables
# into the disjoint union of all the boxes' variables.
function induced_objective(d::AbstractUWD, ps::Vector{MinimizationProblem}, var_map::FinFunction, inclusions::Function)
    proj_mats = Matrix[]
    for b in parts(d, :Box)
        inc = compose(inclusions(b), var_map)
        push!(proj_mats, induced_matrix(inc))
    end
    return z::Vector -> sum([ps[b](proj_mats[b]*z) for b in 1:length(ps)])
end

function induced_objective(d::AbstractUWD, ps::Vector{DualMaxProblem}, var_map::FinFunction, 
    primal_inclusions::Function, dual_inclusions::Function)
    primal_proj_mats = Matrix[]
    dual_proj_mats = Matrix[]
    for b in parts(d, :Box)
        dual_inc = compose(dual_inclusions(b), var_map)
        push!(dual_proj_mats, induced_matrix(dual_inc))
        primal_inc = primal_inclusions(b)
        push!(primal_proj_mats, induced_matrix(primal_inc))
    end
    return (x,λ) ->
        sum([objective(ps[b])(primal_proj_mats[b]*x, dual_proj_mats[b]*λ) for b in 1:length(ps)])
    #return x::Vector -> sum([primal_objective(ps[b])(primal_proj_mats[b]*x) for b in 1:length(ps)]),
    #    z::Vector -> sum([dual_objective(ps[b])(dual_proj_mats[b]*z) for b in 1:length(ps)])
end

""" oapply(d::AbstractUWD, ps::Vector{OpenProblem})

Implements the UWD algebra of open optimization problems. Given a composition pattern (implemented
by an undirected wiring diagram `d`) and open subproblems `ps` returns the composite
open optimization problem.

Each box of `d` must be filled by a problem of the appropriate type signature.
"""
function oapply(d::AbstractUWD, ps::Vector{MinimizationProblem})
    # Check that the number of problems provided matches the number of boxes in the UWD
    nboxes(d) == length(ps) || error("Number of problems does not match number of boxes.")
    # Ensure that each problem fills its associated box
    for i in 1:nboxes(d)
        fills(d, i, ps[i]) || error("Problem $i doesn't fill Box $i")
    end

    M = coproduct((FinSet∘nvars).(ps))
    inclusions(b::Int) = legs(M)[b]

    Mpo = induced_vars(d, ps, inclusions)
    #println(typeof(Mpo))

    obj = induced_objective(d, ps, legs(Mpo)[1], inclusions)

    junction_map = legs(Mpo)[2]
    outer_junction_map = FinFunction(subpart(d, :outer_junction), nparts(d, :Junction))
    return MinimizationProblem(
        length(induced_ports(d)),
        length(apex(Mpo)),
        obj,
        compose(outer_junction_map, junction_map)
    )
end

function oapply(d::AbstractUWD, ps::Vector{DualMaxProblem})
    # Check that the number of problems provided matches the number of boxes in the UWD
    nboxes(d) == length(ps) || error("Number of problems does not match number of boxes.")
    # Ensure that each problem fills its associated box
    for i in 1:nboxes(d)
        fills(d, i, ps[i]) || error("Problem $i doesn't fill Box $i")
    end

    M = coproduct((FinSet∘nvars).(ps))
    dual_inclusions(b::Int) = legs(M)[b]

    N = coproduct((FinSet∘n_primal_vars).(ps))
    primal_inclusions(b::Int) = legs(N)[b]

    Mpo = induced_vars(d, ps, dual_inclusions)
    #println(typeof(Mpo))

    obj = induced_objective(d, ps, legs(Mpo)[1], primal_inclusions, dual_inclusions)

    junction_map = legs(Mpo)[2]
    outer_junction_map = FinFunction(subpart(d, :outer_junction), nparts(d, :Junction))
    return DualMaxProblem(
        length(induced_ports(d)),
        length(apex(Mpo)),
        length(apex(N)),
        obj,
        compose(outer_junction_map, junction_map)
    )
end

# Gradient flow as an algebra morphism
######################################

gradient_flow(p::MinimizationProblem) = ContinuousResourceSharer{Float64}(
        n_exposed_vars(p), 
        nvars(p), 
        (x,_,_) -> -ForwardDiff.gradient(objective(p), x),
        portmap(p).func
)

gradient_flow(p::MinimizationProblem, grad::Function #=Function × R^N->\R^N=#) = ContinuousResourceSharer{Float64}(
    n_exposed_vars(p),
    nvars(p),
    (x,_,_) -> -grad(objective(p), x),
    portmap(p).func
)

function gradient_flow(p::DualMaxProblem)
    x(y) = optimize(primal_objective(p,y), zeros(n_primal_vars(p)), NewtonTrustRegion(), autodiff=:forward).minimizer
    return ContinuousResourceSharer{Float64}(
        n_exposed_vars(p),
        nvars(p),
        (y,_,_) -> ForwardDiff.gradient(dual_objective(p, x(y)), y),
        portmap(p).func
    )
end

#=function gradient_flow(p::DualMaxProblem)
    function primal_grad_flow(λ)
        x -> -ForwardDiff.gradient(primal_objective(p,λ), x)
    end

    return ContinuousResourceSharer{Float64}(
        n_exposed_vars(p),
        nvars(p),
        (λ,_,_) -> ForwardDiff.gradient(dual_objective(p, 
            nlsolve(primal_grad_flow(λ), zeros(n_primal_vars(p))#=, xtol=.01=#).zero), λ),
        portmap(p).func
    )
end=#

#=function gradient_flow(p::DualMaxProblem)
    function primal_grad_flow(λ)
        x -> -ForwardDiff.gradient(primal_objective(p,λ), x)
    end

    return ContinuousResourceSharer{Float64}(
        n_exposed_vars(p),
        nvars(p),
        (λ,_,_) -> ForwardDiff.gradient(dual_objective(p, 
            nlsolve(primal_grad_flow(λ), zeros(n_primal_vars(p))#=, xtol=.01=#).zero), λ),
        portmap(p).func
    )
end=#

gradient_flow(ps::Vector{AbstractProblem}) = map(p->gradient_flow(p), ps)

#=function primal_solution(p::DualMaxProblem, λ)
    function primal_grad_flow(λ)
        x -> -ForwardDiff.gradient(primal_objective(p,λ), x)
    end
    return nlsolve(primal_grad_flow(λ), zeros(n_primal_vars(p))#=, xtol=.01=#)
end=#

function primal_solution(p::DualMaxProblem, y)
    return optimize(primal_objective(p,y), zeros(n_primal_vars(p)), NewtonTrustRegion(), autodiff=:forward).minimizer
end

end # module