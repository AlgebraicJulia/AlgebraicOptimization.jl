""" Define the UWD algebra of open optimization problems
"""
module OpenProblems

export OpenProblem, nvars, n_exposed_vars, portmap, objective, gradient_flow

using Catlab
using AlgebraicDynamics.UWDDynam
using ForwardDiff
import Catlab.oapply

# Open Problems
###############

""" OpenProblem

An open optimization problem R^`vars` -> R. The problem exposes `exposed_vars` variables
given by the portmap `p`. The function `objective` should take `Vector{Float64}` of length
`vars` as input and return a scalar.
"""
struct OpenProblem
    exposed_vars::Int
    vars::Int
    objective::Function
    p::FinFunction
    pint::FinFunction
    OpenProblem(evs, vs, obj, p, pint) = dom(p) != FinSet(evs) || codom(p) != FinSet(vs) ?
        error("Invalid portmap") : new(evs, vs, obj, p, pint)
end
# Convenience constructor when `vars`==`exposed_vars` and `p` is the identity function
OpenProblem(nvars::Int, obj) = OpenProblem(nvars,nvars,obj,FinFunction(1:nvars),FinFunction(1:nvars))
OpenProblem(vvars::Vector{Int}, obj) = OpenProblem(length(vvars),length(vvars),obj,FinFunction(1:length(vvars)),
    FinFunction(vcat([repeat([i],vvars[i]) for i in 1:length(vvars)]...),sum(vvars),length(vvars)))

# Make `OpenProblem`s callable
(p::OpenProblem)(z::Vector) = p.objective(z)

nvars(p::OpenProblem) = p.vars
n_exposed_vars(p::OpenProblem) = p.exposed_vars
objective(p::OpenProblem) = p.objective
portmap(p::OpenProblem) = p.p
internalmap(p::OpenProblem) = p.pint

# Open problems as a UWD algebra
################################

""" fills(d::AbstractUWD, b::Int, p::OpenProblem)

Checks if `p` is of the correct signature to fill box `b` of the uwd `d`
"""
fills(d::AbstractUWD, b::Int, p::OpenProblem) = 
    b <= nparts(d, :Box) ? length(incident(d, b, :box)) == n_exposed_vars(p) : 
        error("Trying to fill box $b when $d has fewer than $b boxes")

### oapply helper functions

induced_ports(d::AbstractUWD) = nparts(d, :OuterPort)
induced_ports(d::RelationDiagram) = subpart(d, [:outer_junction, :variable])

# Returns the pushout which induces the new set of variables
function induced_vars(d::AbstractUWD, ps::Vector{OpenProblem}, inclusions::Function)
    for b in parts(d, :Box)
        fills(d, b, ps[b]) || error("$(ps[b]) does not fill box $b")
    end

    total_portmap = copair([compose(portmap(ps[i]), inclusions(i)) for i in 1:length(ps)])

    #return pushout(FinFunction(subpart(d, :junction), nparts(d, :Junction)), total_portmap)
    return pushout(total_portmap, FinFunction(subpart(d, :junction), nparts(d, :Junction)))
end

#=
# Takes a FinFunction from N->M and returns the induced linear map R^M->R^N
function induced_matrix(dom::Int, codom::Int, f::Vector{Int})::Matrix{Float64}
    length(f) == dom && max(f...) <= codom || error("Invalid FinFunction.")
    res = zeros(dom, codom)
    for (i,j) in Iterators.product(1:dom, 1:codom)
        if f[i] == j
            res[i,j] = 1
        end
    end
    return res
end
=#

function induced_matrix(intmap, f::FinFunction, vmap)::Matrix{Float64}
    # length(f) == dom && max(f...) <= codom || error("Invalid FinFunction.")
    res = zeros(length(dom(intmap)), length(dom(vmap)))

    for k in 1:length(dom(f))
        coords = zip(preimage(intmap,k),preimage(vmap,f(k)))
        for coord in coords
            res[coord[1],coord[2]] = 1
        end
    end
    return res
end

induced_matrix(f::FinFunction) = induced_matrix(length(dom(f)), length(codom(f)), f.func)

# Sums objective functions of `ps` subject to correct variable sharing according to `d`.
# `var_map` should be the left leg of the induced variable pushout.
# `inclusions` should be a function mapping box numbers to the inclusion of that box's variables
# into the disjoint union of all the boxes' variables.
function induced_objective(d::AbstractUWD, ps::Vector{OpenProblem}, var_map::FinFunction, inclusions::Function, vmap)
# function induced_objective(d::AbstractUWD, ps::Vector{OpenProblem}, var_map::FinFunction, inclusions::Function)
    proj_mats = Matrix[]
    for b in parts(d, :Box)
        inc = compose(inclusions(b), var_map)
        # push!(proj_mats, induced_matrix(inc))
        push!(proj_mats, induced_matrix(ps[b].pint,inc,vmap))
    end
    return z -> sum([ps[b](Vector((proj_mats[b]*z)[:])) for b in 1:length(ps)])
end

""" oapply(d::AbstractUWD, ps::Vector{OpenProblem})

Implements the UWD algebra of open optimization problems. Given a composition pattern (implemented
by an undirected wiring diagram `d`) and open subproblems `ps` returns the composite
open optimization problem.

Each box of `d` must be filled by a problem of the appropriate type signature.
"""
function oapply(d::AbstractUWD, ps::Vector{OpenProblem})
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

    total_vportmap = copair([compose(internalmap(ps[i]),portmap(ps[i]), inclusions(i)) for i in 1:length(ps)])
    vmap = FinFunction(vcat([repeat([i],length(preimage(total_vportmap,preimage(legs(Mpo)[1],i)[1]))) for i in 1:length(codom(legs(Mpo)[1]))]...))

    # obj = induced_objective(d, ps, legs(Mpo)[1], inclusions)
    obj = induced_objective(d, ps, legs(Mpo)[1], inclusions,vmap)

    junction_map = legs(Mpo)[2]
    outer_junction_map = FinFunction(subpart(d, :outer_junction), nparts(d, :Junction))
    return OpenProblem(
        length(induced_ports(d)),
        length(apex(Mpo)),
        obj,
        compose(outer_junction_map, junction_map),
        vmap
    )
end

# Gradient flow as an algebra morphism
######################################

gradient_flow(p::OpenProblem) = VectContinuousResourceSharer{Float64}(
        nvars(p), 
        n_exposed_vars(p), 
        # length(dom(p.p)), 
        (x,_,_) -> -ForwardDiff.gradient(objective(p), x),
        portmap(p).func,
        # [preimage(portmap(p),p.pint(i)) for i in dom(p.pint)]
        p.pint
)

gradient_flow(ps::Vector{OpenProblem}) = map(p->gradient_flow(p), ps)

# ContinuousResourceSharer with vect vars
###

import AlgebraicDynamics.UWDDynam: AbstractUndirectedInterface, UndirectedInterface, AbstractUndirectedSystem, ResourceSharer, oapply, induced_states, induced_ports, induced_dynamics, eval_dynamics, euler_approx, nports

struct VectContinuousUndirectedSystem{T} <: AbstractUndirectedSystem{T}
  nstates::Int
  dynamics::Function 
  portmap
  varmap
end
varmap(system::VectContinuousUndirectedSystem) = system.varmap
nstates(system:: VectContinuousUndirectedSystem) = system.nstates 
dynamics(system:: VectContinuousUndirectedSystem) = system.dynamics 
portmap(system:: VectContinuousUndirectedSystem) = system.portmap 
portfunction(system:: VectContinuousUndirectedSystem) = FinFunction(portmap(system), nstates(system))
exposed_states(system:: VectContinuousUndirectedSystem, u::AbstractVector) = getindex(u, portmap(system))

abstract type AbstractVectResourceSharer{T} end

struct VectResourceSharer{T, I, S} <: AbstractVectResourceSharer{T} 
    interface::I 
    system::S
  end
system(r::VectResourceSharer) = r.system 
interface(r::VectResourceSharer) = r.interface
ports(r::VectResourceSharer) = ports(interface(r))
nports(r::VectResourceSharer) = nports(interface(r))
nstates(r::VectResourceSharer) = nstates(system(r)) 
dynamics(r::VectResourceSharer) = dynamics(system(r))
portmap(r::VectResourceSharer) = portmap(system(r)) 
varmap(r::VectResourceSharer) = varmap(system(r)) 
portfunction(r::VectResourceSharer) = portfunction(system(r))
exposed_states(r::VectResourceSharer, u::AbstractVector) = exposed_states(system(r), u)

const VectContinuousResourceSharer{T, I} = VectResourceSharer{T, I, VectContinuousUndirectedSystem}

VectContinuousResourceSharer{T, I}(nports, nstates, dynamics, portmap, varmap) where {T, I <: AbstractUndirectedInterface} =
  VectContinuousResourceSharer{T, I}(UndirectedInterface{T}(nports), VectContinuousUndirectedSystem{T}(nstates, dynamics, portmap, varmap))

VectContinuousResourceSharer{T, N}(nports, nstates, dynamics, portmap, varmap) where {T, N} =
  VectContinuousResourceSharer{T, UndirectedVectorInterface{T}}(nports, nstates, dynamics, portmap, varmap)

VectContinuousResourceSharer{T}(interface::I, system::VectContinuousUndirectedSystem{T}) where {T, I <: AbstractUndirectedInterface} =
  VectContinuousResourceSharer{T, I}(interface, system)

VectContinuousResourceSharer{T}(nports, nstates, dynamics, portmap, varmap) where {T} =
  VectContinuousResourceSharer{T}(UndirectedInterface{T}(nports), VectContinuousUndirectedSystem{T}(nstates, dynamics, portmap, varmap))
            
"""    ContinuousResourceSharer{T}(nstates, f)
If `nports` and `portmap` are not specified by the user, then it is assumed that `nports` is equal to `nstates` and 
`portmap` is the identity map.
"""
VectContinuousResourceSharer{T}(nstates::Int, dynamics::Function) where T =
    VectContinuousResourceSharer{T}(nstates, nstates, dynamics, 1:nstates)


function oapply(d::AbstractUWD, xs::Vector{R}) where {R <: VectContinuousResourceSharer}    
    # S′ = induced_states(d, xs) # ::Pushout
    S = coproduct((FinSet∘nstates).(xs))  
    total_portfunction = copair([compose( portfunction(xs[i]), legs(S)[i]) for i in 1:length(xs)])
    S′ = pushout(total_portfunction, FinFunction(subpart(d, :junction), nparts(d, :Junction)))

    # S = coproduct((FinSet∘nstates).(xs))
    states(b::Int) = legs(S)[b].func

    inclusions(b::Int) = legs(S)[b]
    total_vportmap = copair([compose(varmap(xs[i]),portfunction(xs[i]), inclusions(i)) for i in 1:length(xs)])
    vmap = FinFunction(vcat([repeat([i],length(preimage(total_vportmap,preimage(legs(S′)[1],i)[1]))) for i in 1:length(codom(legs(S′)[1]))]...))

    v = induced_dynamics(d, xs, legs(S′)[1], inclusions, vmap) # states, ) #

    junction_map = legs(S′)[2]
    outer_junction_map = FinFunction(subpart(d, :outer_junction), nparts(d, :Junction))

    return R(
        induced_ports(d), 
        length(apex(S′)), 
        v, 
        compose(outer_junction_map, junction_map).func,
        vmap
    )
end

# 
function induced_dynamics(d::AbstractUWD, xs::Vector{R}, state_map::FinFunction, inclusions::Function, vmap) where {T, R<:VectContinuousResourceSharer{T}}
    proj_mats = Matrix[]
    for b in parts(d, :Box)
        inc = compose(inclusions(b), state_map)
        push!(proj_mats, induced_matrix(varmap(xs[b]),inc,vmap))
    end
      
    function v(u′::AbstractVector, p, t::Real)
      # u = getindex(u′,  state_map.func)
      du = zero(u′)
      # apply dynamics
      for b in parts(d, :Box)
        du += proj_mats[b]'*eval_dynamics(xs[b],Vector((proj_mats[b]*u′)[:]), p, t) 
      end
      # add along junctions
      # du′ = [sum(Array{T}(view(du, preimage(state_map, i)))) for i in codom(state_map)]
      du′ = du
      return du′
    end
end

eval_dynamics(r::AbstractVectResourceSharer, u::AbstractVector, p, t::Real) = dynamics(r)(u, p, t)
eval_dynamics!(du, r::AbstractVectResourceSharer, u::AbstractVector, p, t::Real) = begin
    du .= eval_dynamics(r, u, p, t)
end
eval_dynamics(r::AbstractVectResourceSharer, u::AbstractVector) = eval_dynamics(r, u, [], 0)
eval_dynamics(r::AbstractVectResourceSharer, u::AbstractVector, p) = eval_dynamics(r, u, p, 0)


euler_approx(f::VectContinuousResourceSharer{T}, h::Float64) where T = DiscreteResourceSharer{T}(
        nports(f), nstates(f), 
        (u, p, t) -> u + h*eval_dynamics(f, u, p, t),
        portmap(f)
)

"""    euler_approx(r::ContinuousResourceSharer)

Transforms a continuous resource sharer `r` into a discrete resource sharer via Euler's method.
The step size parameter is appended to the end of the system's parameter list.
"""
euler_approx(f::VectContinuousResourceSharer{T}) where T = DiscreteResourceSharer{T}(
    nports(f), nstates(f), 
    (u, p, t) -> u + p[end]*eval_dynamics(f, u, p[1:end-1], t),
    portmap(f)
)
end # module