#module FinSetAlgebras

using LinearAlgebra, SparseArrays
using Catlab
import Catlab: oapply, dom

#abstract type AlgebraObject end

abstract type FinSetAlgebra{T} end # A finset algebra with object type T

#=function dom(X::T)::FinSet where T <: AlgebraObject
    error("Domain not specified.")
end=#

#=function ob_map(::FinSetAlgebra{T}, N::FinSet)::T where T <: AlgebraObject
    error("Object map not implemented.")
end=#

function hom_map(::FinSetAlgebra{T}, ϕ::FinFunction, X::T)::T where T
    error("Morphism map not implemented.")
end

function laxator(::FinSetAlgebra{T}, Xs::Vector{T})::T where T
    error("Laxator not implemented.")
end

function oapply(A::FinSetAlgebra{T}, ϕ::FinFunction, Xs::Vector{T})::T where T
    return hom_map(A, ϕ, laxator(A, Xs))
end

# Example
function pullback_matrix(f::FinFunction)
    n = length(dom(f))
    sparse(1:n, f.(dom(f)), ones(Int,n), dom(f).n, codom(f).n)
end

pushforward_matrix(f::FinFunction) = pullback_matrix(f)'

#=struct FreeVector <: AlgebraObject
    dim::FinSet
    v::Vector{Float64}
end=#

#dom(v::FreeVector) = v.dim

struct Pushforward <: FinSetAlgebra{Vector{Float64}} end

dom(v::Vector{Float64}) = FinSet(length(v))

hom_map(::Pushforward, ϕ::FinFunction, v::Vector{Float64}) = pushforward_matrix(ϕ)*v


laxator(::Pushforward, Xs::Vector{Vector{Float64}}) = vcat(Xs...)

# UWD algebras from finset algebras
abstract type CospanAlgebra{T} end
#abstract type SimpleCospanAlgebra{T, A<:FinSetAlgebra{T}} <: CospanAlgebra{T} end

function hom_map(::CospanAlgebra{T}, ϕ::Cospan, X::T)::T where T
    error("Morphism map not implemented.")
end

function laxator(::CospanAlgebra{T}, Xs::Vector{T})::T where T
    error("Laxator not implemented.")
end


function oapply(A::CospanAlgebra{T}, ϕ::Cospan, Xs::Vector{T})::T where T
    return hom_map(A, ϕ, laxator(A, Xs))
end

struct Open{T}
    S::FinSet
    o::T
    m::FinFunction    
    Open{T}(S, o, m) where T = S != codom(m) || dom(o) != S ? 
        error("Invalid portmap.") : new(S, o, m)
end

dom(obj::Open{T}) where T = dom(obj.m)

function hom_map(::CospanAlgebra{Open{T}}, A::FinSetAlgebra{T}, ϕ::Cospan, X::Open{T})::Open{T} where T
    l = left(c)
    r = right(c)
    p = pushout(X.m, l)
    pL = legs(p)[1]
    pR = legs(p)[2]
    return Open{T}(apex(p), hom_map(A, pL, X.o), pR∘r)
end

function laxator(::CospanAlgebra{Open{T}}, A::FinSetAlgebra{T}, Xs::Vector{Open{T}})::Open{T} where T
    S = coproduct([X.S for X in Xs])
    inclusions(i::Int) = legs(S)[i]
    m = copair([compose(Xs[i].m, inclusions(i)) for i in 1:length(Xs)])
    o = laxator(A, [X.o for X in Xs])
    return Open{T}(S, o, m)
end


#end