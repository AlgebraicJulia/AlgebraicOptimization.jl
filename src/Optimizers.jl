module Optimizers

export pullback_matrix, pushforward_matrix, Optimizer, OpenContinuousOpt, OpenDiscreteOpt, Euler,
    simulate

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using Catlab
import Catlab: oapply, dom
using SparseArrays

function pullback_matrix(f::FinFunction)
    n = length(dom(f))
    sparse(1:n, f.(dom(f)), ones(Int,n), dom(f).n, codom(f).n)
end

pushforward_matrix(f::FinFunction) = pullback_matrix(f)'

struct Optimizer
    state_space::FinSet
    dynamics::Function # R^ss -> R^ss
end
(s::Optimizer)(x::Vector) = s.dynamics(x)
dom(s::Optimizer) = s.state_space

struct ContinuousOpt <: FinSetAlgebra{Optimizer} end
struct DiscreteOpt <: FinSetAlgebra{Optimizer} end

hom_map(::ContinuousOpt, ϕ::FinFunction, s::Optimizer) = 
    Optimizer(codom(ϕ), x->pushforward_matrix(ϕ)*s(pullback_matrix(ϕ)*x))

hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer) =
    Optimizer(codom(ϕ), x-> begin 
        y = pullback_matrix(ϕ)*x
        return x + pushforward_matrix(ϕ)*(s(y) - y)
    end)

function laxator(::ContinuousOpt, Xs::Vector{Optimizer})
    c = coproduct([dom(X) for X in Xs])
    subsystems = [x -> X(pullback_matrix(l)*x) for (X,l) in zip(Xs, legs(c))]
    function parallel_dynamics(x)
        res = Vector{Vector}(undef, length(Xs)) # Initialize storage for results
        Threads.@threads for i = 1:length(Xs)
            res[i] = subsystems[i](x)
        end
        return vcat(res...)
    end
    return Optimizer(apex(c), parallel_dynamics)
end

laxator(::DiscreteOpt, Xs::Vector{Optimizer}) = laxator(ContinuousOpt(), Xs)

#const OpenOptimizer = Open{Optimizer}
Open{Optimizer}(S::FinSet, v::Function, m::FinFunction) = Open{Optimizer}(S, Optimizer(S, v), m)


struct OpenContinuousOpt <: CospanAlgebra{Open{Optimizer}} end # Turn Dynam into a UWD algebra in one line!
struct OpenDiscreteOpt <: CospanAlgebra{Open{Optimizer}} end

function oapply(C::OpenContinuousOpt, d::AbstractUWD, Xs::Vector{Open{Optimizer}})
    return oapply(C, ContinuousOpt(), d, Xs)
end

function oapply(C::OpenDiscreteOpt, d::AbstractUWD, Xs::Vector{Open{Optimizer}})
    return oapply(C, DiscreteOpt(), d, Xs)
end

function Euler(f::Open{Optimizer}, γ::Float64)
    return Open{Optimizer}(f.S, Optimizer(f.S, x->x+γ*f.o(x)), f.m)
end

function simulate(f::Open{Optimizer}, x0::Vector{Float64}, tsteps::Int)
    res = x0
    for i in 1:tsteps
        res = f.o(res)
    end
    return res
end

end