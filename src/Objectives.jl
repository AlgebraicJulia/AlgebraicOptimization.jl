module Objectives

export PrimalObjective, MinObj, gradient_flow, 
    SaddleObjective, DualComp, primal_solution, dual_objective, primal_objective

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using ..Optimizers
using Catlab
import Catlab: oapply, dom
using ForwardDiff
using Optim
using ComponentArrays


# Primal Minimization Problems and Gradient Descent
###################################################

""" PrimalObjective

An objective defining a minimization problems.
These consist of a finset representing the decision variables and a cost function
on the decision space. Note the cost function should be autodifferentiable by ForwardDiff.jl.
"""
struct PrimalObjective
    decision_space::FinSet
    objective::Function # R^ds -> R NOTE: should be autodifferentiable
end
(p::PrimalObjective)(x) = p.objective(x)    # Removed x::Vector hard typing
dom(p::PrimalObjective) = p.decision_space

"""     MinObj

Finset-algebra implementing composition of minimization problems by variable sharing.
"""
struct MinObj <: FinSetAlgebra{PrimalObjective} end

"""     hom_map(::MinObj, ϕ::FinFunction, p::PrimalObjective)

The morphism map is defined by ϕ ↦ (f ↦ f∘ϕ^*).
"""
hom_map(::MinObj, ϕ::FinFunction, p::PrimalObjective) = 
    PrimalObjective(codom(ϕ), x->p(pullback_function(ϕ, x)))

"""     laxator(::MinObj, Xs::Vector{PrimalObjective})

Takes the "disjoint union" of a collection of primal objectives.
"""
function laxator(::MinObj, Xs::Vector{PrimalObjective})
    c = coproduct([dom(X) for X in Xs])
    subproblems = [x -> X(pullback_function(l)(x)) for (X,l) in zip(Xs, legs(c))]
    objective(x) = sum([sp(x) for sp in subproblems])
    return PrimalObjective(apex(c), objective)
end

Open{PrimalObjective}(S::FinSet, f::Function, m::FinFunction) = 
    Open{PrimalObjective}(S, PrimalObjective(S, f), m)

struct OpenMinObj <: CospanAlgebra{Open{PrimalObjective}} end

function oapply(d::AbstractUWD, Xs::Vector{Open{PrimalObjective}})
    return oapply(OpenMinObj(), MinObj(), d, Xs)
end

"""     gradient_flow(f::Open{PrimalObjective})

Returns the gradient flow optimizer of a given primal objective.
"""
function gradient_flow(f::Open{PrimalObjective})
    function f_wrapper(ca::ComponentArray)
        inputs = [ca[key] for key in keys(ca)]
        f.o(inputs)    # To spread or not to spread?|
    end

    function gradient_descent(x)
        init_conds = ComponentVector(;zip([Symbol(i) for i in eachindex(x)], x)...)
        grad = -ForwardDiff.gradient(f_wrapper, init_conds)
        [grad[key] for key in keys(grad)]
    end

    return Open{Optimizer}(f.S, x -> gradient_descent(x), f.m)

    # return Open{Optimizer}(f.S, x -> -ForwardDiff.gradient(f.o, x), f.m)   # Scalar version
end

function solve(f::Open{PrimalObjective}, x0::Vector{Float64}, ss::Float64, n_steps::Int)
    solver = Euler(gradient_flow(f), ss)
    return simulate(solver, x0, n_steps)
end

# Saddle Problems and Dual Ascent
#################################

struct SaddleObjective
    primal_space::FinSet
    dual_space::FinSet
    objective::Function # x × λ → R
end

# struct SaddleObjective
#     decision_space::FinSet
#     type::decision_space -> Bool
#     objective::Function # x × λ → R
# end




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
        (x,λ) -> p(x, pullback_function(ϕ)(λ)))

# Laxate along both primal and dual variables
function laxator(::DualComp, Xs::Vector{SaddleObjective})
    c1 = coproduct([X.primal_space for X in Xs])
    c2 = coproduct([X.dual_space for X in Xs])
    subproblems = [(x,λ) -> 
        X(pullback_function(l1)(x), pullback_function(l2)(λ)) for (X,l1,l2) in zip(Xs, legs(c1), legs(c2))]
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
                    zeros(n_primal_vars(f)),
                    LBFGS(), autodiff=:forward).minimizer
    return Open{Optimizer}(of.S, 
        λ -> ForwardDiff.gradient(dual_objective(f, x(λ)), λ), of.m)
end

end  # module