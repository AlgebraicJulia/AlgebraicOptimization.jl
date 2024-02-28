module FinSetAlgebras

export FinSetAlgebra, CospanAlgebra, Open, hom_map, laxator, data, portmap

using LinearAlgebra, SparseArrays
using Catlab
import Catlab: oapply, dom, Cospan

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
    Open{T}(S, o, m) where T = 
        S != codom(m) || dom(o) != S ? error("Invalid portmap.") : new(S, o, m)
end

data(obj::Open{T}) where T = obj.o
portmap(obj::Open{T}) where T = obj.m


function Open{T}(o::T) where T
    Open{T}(domain(o), o, id(domain(o)))
end

dom(obj::Open{T}) where T = dom(obj.m)

function hom_map(::CospanAlgebra{Open{T}}, A::FinSetAlgebra{T}, ϕ::Cospan, X::Open{T})::Open{T} where T
    l = left(ϕ)
    r = right(ϕ)
    p = pushout(X.m, l)
    pL = legs(p)[1]
    pR = legs(p)[2]
    return Open{T}(apex(p), hom_map(A, pL, X.o), compose(r,pR))
end

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

function uwd_to_cospan(d::AbstractUWD)
    # Build the left leg
    left_dom = vcat([length(ports(d, i)) for i in boxes(d)])
    left_codom = njunctions(d)

    #println(cp_dom)
    ports_to_junctions = FinFunction[]
    total_portmap = subpart(d, :junction)

    for box in ports.([d], boxes(d))
        push!(ports_to_junctions, FinFunction([total_portmap[p] for p in box], length(box), left_codom))
    end
    #println(ports_to_junctions)
    #cp = CompositionPattern(cp_dom, cp_codom, ports_to_junctions)

    left = copair(ports_to_junctions)
    right = FinFunction(subpart(d, :outer_junction), left_codom)
    
    return Cospan(left, right)  
end

function oapply(CA::CospanAlgebra{Open{T}}, FA::FinSetAlgebra{T}, d::AbstractUWD, Xs::Vector{Open{T}})::Open{T} where T
    return oapply(CA, FA, uwd_to_cospan(d), Xs)
end


end
#=
# Test example
struct UWDPushforward <: CospanAlgebra{Open{Vector{Float64}}} end

const OpenVector = Open{Vector{Float64}}

# UWD Interop
function uwd_to_cospan(d::AbstractUWD)
    # Build the left leg
    left_dom = vcat([length(ports(d, i)) for i in boxes(d)])
    left_codom = njunctions(d)

    #println(cp_dom)
    ports_to_junctions = FinFunction[]
    total_portmap = subpart(d, :junction)

    for box in ports.([d], boxes(d))
        push!(ports_to_junctions, FinFunction([total_portmap[p] for p in box], length(box), left_codom))
    end
    #println(ports_to_junctions)
    #cp = CompositionPattern(cp_dom, cp_codom, ports_to_junctions)

    left = copair(ports_to_junctions)
    right = FinFunction(subpart(d, :outer_junction), left_codom)
    
    return Cospan(left, right)  
end

function oapply(CA::CospanAlgebra{Open{T}}, FA::FinSetAlgebra{T}, d::AbstractUWD, Xs::Vector{Open{T}})::Open{T} where T
    return oapply(CA, FA, uwd_to_cospan(d), Xs)
end

# AlgebraicDynamics
struct System
    state_space::FinSet
    dynamics::Function # R^ss -> R^ss
end
(s::System)(x::Vector) = s.dynamics(x)
dom(s::System) = s.state_space

struct Dynam <: FinSetAlgebra{System} end

hom_map(::Dynam, ϕ::FinFunction, s::System) = 
    System(codom(ϕ), x->pushforward_matrix(ϕ)*s(pullback_matrix(ϕ)*x))

function laxator(::Dynam, Xs::Vector{System})
    c = coproduct([dom(X) for X in Xs])
    subsystems = [x -> X(pullback_matrix(l)*x) for (X,l) in zip(Xs, legs(c))]
    function parallel_dynamics(x)
        res = Vector{Vector}(undef, length(Xs)) # Initialize storage for results
        Threads.@threads for i = 1:length(Xs)
            res[i] = subsystems[i](x)
        end
        return vcat(res...)
    end
    return System(apex(c), parallel_dynamics)
end

struct UWDDynam <: CospanAlgebra{Open{System}} end # Turn Dynam into a UWD algebra in one line!

A = rand(-1.0:.01:1.0, 5,5)
B = rand(-1.0:.01:1.0, 3,3)
C = rand(-1.0:.01:1.0, 4,4)

γ = 0.1
s1 = System(FinSet(5), x->x +γ*A*x)
s2 = System(FinSet(3), x->x + γ*B*x)
s3 = System(FinSet(4), x->x + γ*C*x)

ϕ = FinFunction([1,2,3,4,5,2,3,6,3,6,7,8])

s = oapply(Dynam(), ϕ, [s1,s2,s3])=#

#end