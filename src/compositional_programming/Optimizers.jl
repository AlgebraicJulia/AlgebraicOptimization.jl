# Implement the cospan-algebra of dynamical systems.
module Optimizers

export pullback_matrix, pushforward_matrix, Optimizer, OpenContinuousOpt, OpenDiscreteOpt, Euler,
    simulate

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using Catlab
import Catlab: oapply, dom
using SparseArrays

"""     pullback_matrix(f::FinFunction)

The pullback of f : n → m is the linear map f^* : Rᵐ → Rⁿ defined by
f^*(y)[i] = y[f(i)].
"""
function pullback_matrix(f::FinFunction)
    n = length(dom(f))
    sparse(1:n, f.(dom(f)), ones(Int,n), dom(f).n, codom(f).n)
end

"""     pushforward_matrix(f::FinFunction)

The pushforward is the dual of the pullback.
"""
pushforward_matrix(f::FinFunction) = pullback_matrix(f)'

""" Optimizer

An optimizer is defined by a finite set N representing the state space
and a function Rᴺ → Rᴺ representing the dynamics.
"""
struct Optimizer
    state_space::FinSet
    dynamics::Function # R^ss -> R^ss
end
(s::Optimizer)(x::Vector) = s.dynamics(x)
dom(s::Optimizer) = s.state_space

# Finset-algebras for composing continuous and discrete systems.
struct ContinuousOpt <: FinSetAlgebra{Optimizer} end
struct DiscreteOpt <: FinSetAlgebra{Optimizer} end

"""     hom_map(::ContinuousOpt, ϕ::FinFunction, s::Optimizer)

The hom map is defined as ϕ ↦ (s ↦ ϕ_*∘s∘ϕ^*).
"""
hom_map(::ContinuousOpt, ϕ::FinFunction, s::Optimizer) = 
    Optimizer(codom(ϕ), x->pushforward_matrix(ϕ)*s(pullback_matrix(ϕ)*x))

"""     hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer)

The hom map is defined as ϕ ↦ (s ↦ id + ϕ_*∘(s - id)∘ϕ^*).
"""
hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer) =
    Optimizer(codom(ϕ), x-> begin 
        y = pullback_matrix(ϕ)*x
        return x + pushforward_matrix(ϕ)*(s(y) - y)
    end)

"""     laxator(::ContinuousOpt, Xs::Vector{Optimizer})

Takes the "disjoint union" of a collection of optimizers.
"""
function laxator(::ContinuousOpt, Xs::Vector{Optimizer})
    c = coproduct([dom(X) for X in Xs])
    subsystems = [x -> X(pullback_matrix(l)*x) for (X,l) in zip(Xs, legs(c))]
    function parallel_dynamics(x)
        res = Vector{Vector}(undef, length(Xs)) # Initialize storage for results
        #=Threads.@threads=# for i = 1:length(Xs)
            res[i] = subsystems[i](x)
        end
        return vcat(res...)
    end
    return Optimizer(apex(c), parallel_dynamics)
end
# Same as continuous opt
laxator(::DiscreteOpt, Xs::Vector{Optimizer}) = laxator(ContinuousOpt(), Xs)

Open{Optimizer}(S::FinSet, v::Function, m::FinFunction) = Open{Optimizer}(S, Optimizer(S, v), m)

# Turn into cospan-algebras.
struct OpenContinuousOpt <: CospanAlgebra{Open{Optimizer}} end
struct OpenDiscreteOpt <: CospanAlgebra{Open{Optimizer}} end

function oapply(C::OpenContinuousOpt, d::AbstractUWD, Xs::Vector{Open{Optimizer}})
    return oapply(C, ContinuousOpt(), d, Xs)
end

function oapply(C::OpenDiscreteOpt, d::AbstractUWD, Xs::Vector{Open{Optimizer}})
    return oapply(C, DiscreteOpt(), d, Xs)
end

# Euler's method is a natural transformation from continous optimizers to discrete optimizers.
function Euler(f::Open{Optimizer}, γ::Float64)
    return Open{Optimizer}(f.S, Optimizer(f.S, x->x+γ*f.o(x)), f.m)
end

# Run a discrete optimizer the designated number of time-steps.
function simulate(f::Open{Optimizer}, x0::Vector{Float64}, tsteps::Int)
    res = x0
    for i in 1:tsteps
        res = f.o(res)
    end
    return res
end

end