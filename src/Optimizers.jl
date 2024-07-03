# Implement the cospan-algebra of dynamical systems.
module Optimizers

export pullback_matrix, pushforward_matrix, Optimizer, OpenContinuousOpt, OpenDiscreteOpt, Euler,
    simulate, typed_pullback_matrix, test_pullback_function, test_pushforward_function

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
    sparse(1:n, f.(dom(f)), ones(Int, n), dom(f).n, codom(f).n)
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
    Optimizer(codom(ϕ), x -> test_pushforward_function(ϕ, s(test_pullback_function(ϕ, x))))

"""     hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer)

The hom map is defined as ϕ ↦ (s ↦ id + ϕ_*∘(s - id)∘ϕ^*).
"""
hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer) =
    Optimizer(codom(ϕ), x -> begin
        y = test_pullback_function(ϕ, x)
        return x + test_pushforward_function(ϕ, (s(y) - y))
    end)

"""     laxator(::ContinuousOpt, Xs::Vector{Optimizer})

Takes the "disjoint union" of a collection of optimizers.
"""
function laxator(::ContinuousOpt, Xs::Vector{Optimizer})
    c = coproduct([dom(X) for X in Xs])
    subsystems = [x -> X(test_pullback_function(l, x)) for (X, l) in zip(Xs, legs(c))]
    function parallel_dynamics(x)
        res = Vector{Vector}(undef, length(Xs)) # Initialize storage for results
        for i = 1:length(Xs)        #=Threads.@threads=#
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
    return Open{Optimizer}(f.S,
     Optimizer(f.S, x -> begin println(x); println(f.o(x)); println("hey there buddy"); println(f.o.dynamics);
     x .+ γ .* f.o(x)
    end), f.m)
end

# Run a discrete optimizer the designated number of time-steps.
function simulate(f::Open{Optimizer}, x0::Vector{Float64}, tsteps::Int)
    res = x0
    for i in 1:tsteps
        res = f.o(res)
    end
    return res
end


function simulate(f::Open{Optimizer}, x0::Vector{Vector{Float64}}, tsteps::Int)
    res = x0
    for i in 1:tsteps
        res = f.o(res)
    end
    return res
end






function typed_pullback_matrix(f, domType, codomType)  # Modify code
    # Track codomain indices
    prefixes = Dict()
    lastPrefix = 0

    for v in codom(f)    # Assumes codom(f) is a set and has distinct elements
        prefixes[v] = lastPrefix
        lastPrefix += codomType(v)
    end
    # lastPrefix now holds the sum of the sizes across all the output vectors

    domLength = 0
    result = []
    for v in dom(f)
        domLength += domType(v)
        for i in 1:domType(v)
            push!(result, prefixes[f(v)] + i)
        end
    end

    sparse(1:domLength, result, ones(Int, domLength), domLength, lastPrefix)
end



function typed_pullback_matrix(f)  # No types provided, so assume everything uses scalars

    # println("inside typed")

    domType = FinFunction(ones(Int, length(dom(f))))
    codomType = FinFunction(ones(Int, length(codom(f))))

    # Track codomain indices
    prefixes = Dict()
    lastPrefix = 0

    for v in codom(f)    # Assumes codom(f) is a set and has distinct elements
        prefixes[v] = lastPrefix
        lastPrefix += codomType(v)
    end
    # lastPrefix now holds the sum of the sizes across all the output vectors

    domLength = 0
    result = []
    for v in dom(f)
        domLength += domType(v)
        for i in 1:domType(v)
            push!(result, prefixes[f(v)] + i)
        end
    end

    sparse(1:domLength, result, ones(Int, domLength), domLength, lastPrefix)
end


typed_pushforward_matrix(f::FinFunction) = typed_pullback_matrix(f)'
typed_pushforward_matrix(f::FinFunction, domType, codomType) = typed_pullback_matrix(f, domType, codomType)'



function test_pullback_function(f::FinFunction, v::Vector)::Vector
    return [v[f(i)] for i in 1:length(dom(f))]
end

function test_pushforward_function(f::FinFunction, v)::Vector
    # output = zeros(length(codom(f)))
    output = [[] for _ in 1:length(codom(f))]


    # output = Vector{Vector}(nothing, length(codom(f)))

    # println(output)

    for i in 1:length(dom(f))
        # println("i = ", i)
        # println(output)
        # println(output[f(i)])
        # println()
        if isempty(output[f(i)])
            output[f(i)] = v[i]
        else
            output[f(i)] += v[i]
        end
    end

    return output
end


function test_pushforward_function(f::FinFunction, v::Vector{Float64})::Vector
    # output = zeros(length(codom(f)))
    output = [0.0 for _ in 1:length(codom(f))]


    # output = Vector{Vector}(nothing, length(codom(f)))

    # println(output)

    for i in 1:length(dom(f))
        # println("i = ", i)
        # println(output)
        # println(output[f(i)])
        # println()
        output[f(i)] += v[i]
    end

    return output
end











end  # module




