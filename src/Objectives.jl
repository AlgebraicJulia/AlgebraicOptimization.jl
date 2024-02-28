module Objectives

export PrimalObjective, MinObj, gradient_flow, 
    SaddleObjective, DualComp, primal_solution

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using ..Optimizers
using Catlab
import Catlab: oapply, dom
using ForwardDiff
using Optim

# Primal Minimization Problems and Gradient Descent
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
    return Open{Optimizer}(f.S, x -> -ForwardDiff.gradient(f.o, x), f.m)
end

# Saddle Problems and Dual Ascent
struct SaddleObjective
    primal_space::FinSet
    dual_space::FinSet
    objective::Function # x × λ → R
end

(p::SaddleObjective)(x,λ) = p.objective(x,λ)

n_primal_vars(p::SaddleObjective) = length(p.primal_space)
dom(p::SaddleObjective) = p.dual_space
objective(p::SaddleObjective) = p.objective
primal_objective(p::SaddleObjective, λ) = 
    x -> objective(p)(x,λ)
dual_objective(p::SaddleObjective, x) = 
    λ -> objective(p)(x,λ) 

primal_solution(p::SaddleObjective, λ) =
    optimize(primal_objective(p,λ), zeros(n_primal_vars(p)), LBFGS(), autodiff=:forward).minimizer

# finset algebra for composing along dual variables of saddle functions
struct DualComp <: FinSetAlgebra{SaddleObjective} end

# Only "glue" along dual variables
hom_map(::DualComp, ϕ::FinFunction, p::SaddleObjective) = 
    SaddleObjective(p.primal_space, codom(ϕ), 
        (x,λ) -> p(x, pullback_matrix(ϕ)*λ))

# Laxate along both primal and dual variables
function laxator(::DualComp, Xs::Vector{SaddleObjective})
    c1 = coproduct([X.primal_space for X in Xs])
    c2 = coproduct([X.dual_space for X in Xs])
    subproblems = [(x,λ) -> 
        X(pullback_matrix(l1)*x, pullback_matrix(l2)*λ) for (X,l1,l2) in zip(Xs, legs(c1), legs(c2))]
    objective(x,λ) = sum([sp(x,λ) for sp in subproblems])
    return SaddleObjective(apex(c1), apex(c2), objective)
end

struct OpenDualComp <: CospanAlgebra{Open{SaddleObjective}} end

function oapply(d::AbstractUWD, Xs::Vector{Open{SaddleObjective}})
    return oapply(OpenDualComp(), DualComp(), d, Xs)
end

function gradient_flow(of::Open{SaddleObjective})
    f = data(of)
    x(λ) = optimize(primal_objective(f,λ), 
                    zeros(length(f.primal_space)),
                    LBFGS(), autodiff=:forward).minimizer
    return Open{Optimizer}(of.S, 
        λ -> ForwardDiff.gradient(dual_objective(f, x(λ)), λ), of.m)
end

end