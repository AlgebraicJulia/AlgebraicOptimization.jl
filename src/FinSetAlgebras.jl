# A module for turning FinSet-algebras into Cospan-algebras
# TODO: upstream into Catlab.jl
module FinSetAlgebras

export FinSetAlgebra, CospanAlgebra, Open, hom_map, laxator, data, portmap

using LinearAlgebra, SparseArrays
using Catlab
import Catlab: oapply, dom, Cospan

"""     FinSetAlgebra{T}

A finset algebra is a lax symmetric monoidal functor (FinSet,+) → (Set,×).
We implicitly use the category of Julia types and (pure) functions as a model of Set,
so T is the type of objects mapped to by the algebra. Finset algebras must then implement
the hom_map and laxator methods.

T must implement dom(x::T)::FinSet, which implicitly defines the object map of a finset-algebra.
"""
abstract type FinSetAlgebra{T} end

"""     hom_map(::FinSetAlgebra{T}, ϕ::FinFunction, X::T)::T where T

Overload to implement the action of a finset-algebra on morphisms.
"""
function hom_map(::FinSetAlgebra{T}, ϕ::FinFunction, X::T)::T where T
    error("Morphism map not implemented.")
end

"""     laxator(::FinSetAlgebra{T}, Xs::Vector{T})::T where T

Overload to implement the product comparison (aka laxator) of a finset algebra.
"""
function laxator(::FinSetAlgebra{T}, Xs::Vector{T})::T where T
    error("Laxator not implemented.")
end

"""     oapply(A::FinSetAlgebra{T}, ϕ::FinFunction, Xs::Vector{T})::T where T

Implements operadic composition for a given finset-algebra implementation.
"""
function oapply(A::FinSetAlgebra{T}, ϕ::FinFunction, Xs::Vector{T})::T where T
    return hom_map(A, ϕ, laxator(A, Xs))
end

# UWD-algebras (aka Cospan-algebras) from finset-algebras
#########################################################

"""     CospanAlgebra{T}

A cospan-algebra is a lax symmetric monoidal functor (Cospan(FinSet),+) → (Set,×).
"""
abstract type CospanAlgebra{T} end

function hom_map(::CospanAlgebra{T}, ϕ::Cospan, X::T)::T where T
    error("Morphism map not implemented.")
end

function laxator(::CospanAlgebra{T}, Xs::Vector{T})::T where T
    error("Laxator not implemented.")
end

function oapply(A::CospanAlgebra{T}, ϕ::Cospan, Xs::Vector{T})::T where T
    return hom_map(A, ϕ, laxator(A, Xs))
end

"""     Open{T}

Given a type T which implements finset-algebra, Open{T} implements cospan-algebra.
o::T is an object, S is the domain of o, and m : dom(m) → S specifies which parts of
S are open for composition.
"""
struct Open{T}
    S::FinSet
    o::T
    m::FinFunction    
    Open{T}(S, o, m) where T = 
        S != codom(m) || dom(o) != S ? error("Invalid portmap.") : new(S, o, m)
end

# Getters for Open{T}
data(obj::Open{T}) where T = obj.o
portmap(obj::Open{T}) where T = obj.m

# Helper function for when m is identity.
function Open{T}(o::T) where T
    Open{T}(domain(o), o, id(domain(o)))
end

function Open{T}(o::T, m::FinFunction) where T
    Open{T}(domain(o), o, m)
end

dom(obj::Open{T}) where T = dom(obj.m)

# Implement the hom_map for a cospan-algebra based on the hom map for a finset-algebra.
function hom_map(::CospanAlgebra{Open{T}}, A::FinSetAlgebra{T}, ϕ::Cospan, X::Open{T})::Open{T} where T
    l = left(ϕ)
    r = right(ϕ)
    p = pushout(X.m, l)
    pL = legs(p)[1]
    pR = legs(p)[2]
    return Open{T}(apex(p), hom_map(A, pL, X.o), compose(r,pR))
end

# Implement the laxator for a cospan-algebra based on the laxator of a finset-algebra.
function laxator(::CospanAlgebra{Open{T}}, A::FinSetAlgebra{T}, Xs::Vector{Open{T}})::Open{T} where T
    S = coproduct([X.S for X in Xs])
    inclusions(i::Int) = legs(S)[i]
    m = copair([compose(Xs[i].m, inclusions(i)) for i in 1:length(Xs)])
    o = laxator(A, [X.o for X in Xs])
    return Open{T}(apex(S), o, m)
end

function oapply(CA::CospanAlgebra{Open{T}}, FA::FinSetAlgebra{T}, ϕ::Cospan, Xs::Vector{Open{T}})::Open{T} where T
    return hom_map(CA, FA, ϕ, laxator(CA, FA, Xs))
end

"""     uwd_to_cospan(d::AbstractUWD)

Returns the underlying cospan representation of a given UWD.
"""
function uwd_to_cospan(d::AbstractUWD)
    # Build the left leg
    left_dom = vcat([length(ports(d, i)) for i in boxes(d)])
    left_codom = njunctions(d)

    ports_to_junctions = FinFunction[]
    total_portmap = subpart(d, :junction)

    for box in ports.([d], boxes(d))
        push!(ports_to_junctions, FinFunction([total_portmap[p] for p in box], length(box), left_codom))
    end

    left = copair(ports_to_junctions)
    right = FinFunction(subpart(d, :outer_junction), left_codom)
    
    return Cospan(left, right)  
end

function oapply(CA::CospanAlgebra{Open{T}}, FA::FinSetAlgebra{T}, d::AbstractUWD, Xs::Vector{Open{T}})::Open{T} where T
    return oapply(CA, FA, uwd_to_cospan(d), Xs)
end

end