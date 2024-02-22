module Objectives

export PrimalObjective, MinObj, gradient_flow

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using ..Optimizers
using Catlab
import Catlab: oapply, dom
using ForwardDiff

struct PrimalObjective
    decision_space::FinSet
    objective::Function # R^ds -> R NOTE: should be autodifferentiable
end
(p::PrimalObjective)(x::Vector) = p.objective(x)
dom(p::PrimalObjective) = p.decision_space

struct MinObj <: FinSetAlgebra{PrimalObjective} end

hom_map(::MinObj, ϕ::FinFunction, p::PrimalObjective) = 
    PrimalObjective(codom(ϕ), x->p(pullback_matrix(ϕ)*x))

function laxator(::MinObj, Xs::Vector{PrimalObjective})
    c = coproduct([dom(X) for X in Xs])
    subproblems = [x -> X(pullback_matrix(l)*x) for (X,l) in zip(Xs, legs(c))]
    objective(x) = sum([sp(x) for sp in subproblems])
    return PrimalObjective(apex(c), objective)
end

Open{PrimalObjective}(S::FinSet, f::Function, m::FinFunction) = 
    Open{PrimalObjective}(S, PrimalObjective(S, f), m)

struct OpenMinObj <: CospanAlgebra{Open{PrimalObjective}} end

function oapply(d::AbstractUWD, Xs::Vector{Open{PrimalObjective}})
    return oapply(OpenMinObj(), MinObj(), d, Xs)
end

function gradient_flow(f::Open{PrimalObjective})
    return OpenOptimizer(f.S, x -> -ForwardDiff.gradient(f.o, x), f.m)
end




end