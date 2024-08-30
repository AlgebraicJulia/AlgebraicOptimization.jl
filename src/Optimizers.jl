# Implement the cospan-algebra of dynamical systems.
module Optimizers

export pullback_matrix, pushforward_matrix, Optimizer, OpenContinuousOpt, OpenDiscreteOpt, Euler,
    simulate, pullback_function, pushforward_function

using ..FinSetAlgebras
import ..FinSetAlgebras: hom_map, laxator
using Catlab
import Catlab: oapply, dom
using ComponentArrays



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
function hom_map(::ContinuousOpt, ϕ::FinFunction, s::Optimizer) 
    Optimizer(codom(ϕ), x -> pushforward_function(ϕ)(s(pullback_function(ϕ)(x))))
end

"""     hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer)

The hom map is defined as ϕ ↦ (s ↦ id + ϕ_*∘(s - id)∘ϕ^*).
"""
hom_map(::DiscreteOpt, ϕ::FinFunction, s::Optimizer) =
    Optimizer(codom(ϕ), x -> begin
        y = pullback_function(ϕ)(x)
        return x + pushforward_function(ϕ)((s(y) - y))
    end)

"""     laxator(::ContinuousOpt, Xs::Vector{Optimizer})

Takes the "disjoint union" of a collection of optimizers.
"""
function laxator(::ContinuousOpt, Xs::Vector{Optimizer})
    c = coproduct([dom(X) for X in Xs])
    subsystems = [x -> X(pullback_function(l)(x)) for (X, l) in zip(Xs, legs(c))]
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
Open{Optimizer}(s::Int, v::Function, m::FinFunction) = Open{Optimizer}(FinSet(s), v, m)

# Special cases: m is an identity
Open{Optimizer}(S::FinSet, v::Function) = Open{Optimizer}(S, Optimizer(S, v), id(S))
Open{Optimizer}(s::Int, v::Function) = Open{Optimizer}(FinSet(s), v)


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
     Optimizer(f.S, x -> x .+ γ .* f.o(x)), f.m)
end

# Run a discrete optimizer the designated number of time-steps.
function simulate(f::Open{Optimizer}, x0::Vector, tsteps::Int)
    res = x0
    for i in 1:tsteps
        res = f.o(res)
    end
    return res
end

# Run a discrete optimizer the designated number of time-steps.
function simulate(f::Open{Optimizer}, d::AbstractUWD, x0::ComponentArray, tsteps::Int)
    # Format initial conditions
    initial_cond_vec = Vector{Any}(undef, length(d[:variable]))
    var_to_index = Dict()
    curr_index = 1
    for junction in d[:junction]
        if !haskey(var_to_index, d[:variable][junction])
            var_to_index[d[:variable][junction]] = curr_index
            curr_index += 1
        end
    end

    for (var, index) in var_to_index
        initial_cond_vec[index] = x0[var]
    end
    res = initial_cond_vec
    # Simulate
    for i in 1:tsteps
        res = f.o(res)
    end

    res_formatted = copy(x0)

    # Rebuild component array
    for (var, index) in var_to_index
        res_formatted[var] = res[index]
    end
    return res_formatted
end

function (f::Open{Optimizer})(x0::Vector)
    return f.o(x0)
end



"""     pullback_function(f::FinFunction, v::Vector)

The pullback of f : n → m is the linear map f^* : Rᵐ → Rⁿ defined by
f^*(y)[i] = y[f(i)].

This is the uncurried version.
"""
function pullback_function(f::FinFunction, v::Vector)::Vector
    return [v[f(i)] for i in 1:length(dom(f))]
end


"""     pullback_function(f::FinFunction)

The pullback of f : n → m is the linear map f^* : Rᵐ → Rⁿ defined by
f^*(y)[i] = y[f(i)].

This is the curried version.
"""
function pullback_function(f::FinFunction)
    return v -> pullback_function(f, v)
end


"""     pushforward_function(f::FinFunction, v::Vector{Vector{Float64}})

The pushforward of f : n → m is the linear map f_* : Rⁿ → Rᵐ defined by
f_*(y)[j] =  ∑ y[i] for i ∈ f⁻¹(j).

This is the uncurried version for vectors of vectors.
"""
function pushforward_function(f::FinFunction, v::Vector{Vector{Float64}})::Vector
    output = [[] for _ in 1:length(codom(f))]
    for i in 1:length(dom(f))
        if isempty(output[f(i)])
            output[f(i)] = v[i]
        else
            output[f(i)] += v[i]
        end
    end
    return output
end


"""     pushforward_function(f::FinFunction, v::Vector{Float64})

The pushforward of f : n → m is the linear map f_* : Rⁿ → Rᵐ defined by
f_*(y)[j] =  ∑ y[i] for i ∈ f⁻¹(j).

This is the uncurried version for vectors of floats.
"""
function pushforward_function(f::FinFunction, v::Vector{Float64})::Vector
    output = [0.0 for _ in 1:length(codom(f))]

    for i in 1:length(dom(f))
        output[f(i)] += v[i]
    end

    return output
end

"""     pushforward_function(f::FinFunction)

The pushforward of f : n → m is the linear map f_* : Rⁿ → Rᵐ defined by
f_*(y)[j] =  ∑ y[i] for i ∈ f⁻¹(j).

This is the curried version.
"""
function pushforward_function(f::FinFunction)
    return v -> pushforward_function(f, v)
end

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


end  # module