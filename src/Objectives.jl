module Objectives

export PrimalObjective, MinObj, gradient_flow, 
    SaddleObjective, DualComp, primal_solution, dual_objective, primal_objective, pullback_function

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using ..Optimizers
using Catlab
import Catlab: oapply, dom
using ForwardDiff
using Optim

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
(p::PrimalObjective)(x::Vector) = p.objective(x)
dom(p::PrimalObjective) = p.decision_space

"""     MinObj

Finset-algebra implementing composition of minimization problems by variable sharing.
"""
struct MinObj <: FinSetAlgebra{PrimalObjective} end

"""     hom_map(::MinObj, ϕ::FinFunction, p::PrimalObjective)

The morphism map is defined by ϕ ↦ (f ↦ f∘ϕ^*).
"""
hom_map(::MinObj, ϕ::FinFunction, p::PrimalObjective) = 
    PrimalObjective(codom(ϕ), x->p(test_pullback_function(ϕ, x)))

"""     laxator(::MinObj, Xs::Vector{PrimalObjective})

Takes the "disjoint union" of a collection of primal objectives.
"""
function laxator(::MinObj, Xs::Vector{PrimalObjective})
    c = coproduct([dom(X) for X in Xs])
    subproblems = [x -> X(test_pullback_function(l, x)) for (X,l) in zip(Xs, legs(c))]
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
    return Open{Optimizer}(f.S, x -> -ForwardDiff.gradient(f.o, x), f.m)
end

# Saddle Problems and Dual Ascent
#################################

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
        (x,λ) -> p(x, test_pullback_function(ϕ, λ)))

# Laxate along both primal and dual variables
function laxator(::DualComp, Xs::Vector{SaddleObjective})
    c1 = coproduct([X.primal_space for X in Xs])
    c2 = coproduct([X.dual_space for X in Xs])
    subproblems = [(x,λ) -> 
        X(test_pullback_function(l1, x), test_pullback_function(l2, λ)) for (X,l1,l2) in zip(Xs, legs(c1), legs(c2))]
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






# New stuff

struct UnivTypedPrimalObjective
    decision_space::FinSet
    objective::Function # R^ds -> R NOTE: should be autodifferentiable
    type::FinDomFunction # R^ds -> Z+, for our use case. One function to rule them all--this will stay the same across our finsets.  FinDomFunction
end

(p::UnivTypedPrimalObjective)(x::Vector) = p.objective(x)
dom(p::UnivTypedPrimalObjective) = p.decision_space

"""     MinObj

# Finset-algebra implementing composition of minimization problems by variable sharing.
# """
struct MinObj <: FinSetAlgebra{PrimalObjective} end

"""     hom_map(::MinObj, ϕ::FinFunction, p::PrimalObjective)

The morphism map is defined by ϕ ↦ (f ↦ f∘ϕ^*).
"""
hom_map(::MinObj, ϕ::FinFunction, p::UnivTypedPrimalObjective) =       # Another version, which wouldn't require a universal type function, would have you pass in a custom type function for your set M. This would require more work in the laxator to take the disjoint union of type functions.
    all(p.type(x) == p.type(ϕ(x)) for x in dom(p)) ?
    UnivTypedPrimalObjective(codom(ϕ), x -> p(test_pullback_function(ϕ, x)), p.type) :
    error("The ϕ provided is not type-preserving.")   # throw an error



"""     laxator(::MinObj, Xs::Vector{PrimalObjective})

Takes the "disjoint union" of a collection of primal objectives.
"""
function laxator(::MinObj, Xs::Vector{UnivTypedPrimalObjective})
    c = coproduct([dom(X) for X in Xs])
    subproblems = [x -> X(test_pullback_function(l, x)) for (X, l) in zip(Xs, legs(c))]
    objective(x) = sum([sp(x) for sp in subproblems])
    return UnivTypedPrimalObjective(apex(c), objective, Xs[1].type)    # Assuming all have the same type function
end






struct TypedPrimalObjective   # Should we put restrictions on the constructor, i.e. check that dom(objective) = dom(type) = decision_space?
    decision_space::FinSet
    objective::Function # R^ds -> R NOTE: should be autodifferentiable. Make it a FinDomFunction?
    type::FinDomFunction
end

(p::TypedPrimalObjective)(x::Vector) = p.objective(x)
dom(p::TypedPrimalObjective) = p.decision_space


hom_map(::MinObj, ϕ::FinFunction, σ::FinDomFunction, τ::FinDomFunction, p::TypedPrimalObjective) =       # τ seems completely unnecessary to me
    all(p.type(x) == σ(ϕ(x)) && p.type(x) == τ(x) for x in dom(p)) ?
    UnivTypedPrimalObjective(codom(ϕ), x -> p(test_pullback_function(ϕ, x)), σ) :   # Note: we didn't check but σ must be applicable across all of codom(ϕ)
    nothing   # throw an error



function laxator(::MinObj, Xs::Vector{TypedPrimalObjective})
    combinedType = copair([X.type for X in Xs])
    c = dom(combinedType)   # c is the coproduct of the decision spaces
    objective(x) = sum(X(test_pullback_function(l, x)) for (X, l) in zip(Xs, legs(c)))  # Calculated the same as before (but simplified onto one line)
    return UnivTypedPrimalObjective(apex(c), objective, combinedType)
end




# Pullback function for a given ϕ and vector v
# function pullback(ϕ::FinFunction, v::Vector)
#     output = Vector{eltype(v)}(undef, length(dom(ϕ)))
#     for i in 1:length(dom(ϕ))
#         output[i] = v[ϕ(i)]
#     end
#     return output
# end

# Optional easier version:   return Vector{eltype(v)}(v[ϕ(i)] for i in 1:length(dom(ϕ)))



# Curried version of the pullback function
function curried_pullback(ϕ::FinFunction)
    return function (v::Vector)
        output = Vector{eltype(v)}(undef, length(dom(ϕ)))
        for i in 1:length(dom(ϕ))
            output[i] = v[ϕ(i)]
        end
        return output
    end
end



function pullback_function(ϕ::FinFunction, v::Vector)
    return v[ϕ.(1:length(dom(ϕ)))]  # Broadcasting with vector of indices
  end




# Not the active one
function typed_pullback_matrix(f::FinFunction)  # Modify code

    # Track codomain indices
    prefixes = Dict()
    lastPrefix = 0

    for v in codom(f)
        prefixes[v] = lastPrefix
        lastPrefix += length(v)
    end
    # lastPrefix now holds the sum of the sizes across all the output vectors

    domLength = 0
    result = []
    for v in dom(f)
        domLength += length(v)
        for i in 1:length(v)
            push!(result, prefixes[f(v)] + i)
        end
    end

    sparse(1:domLength, result, ones(Int, domLength), domLength, lastPrefix)
end








end  # module