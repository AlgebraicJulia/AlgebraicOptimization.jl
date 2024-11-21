using LinearAlgebra
using Test

# Compute the pullback of the diagram
#  l - A -> m <- B - n
# in the category of matrices.
function pullback(A::AbstractMatrix, B::AbstractMatrix)
    @assert size(A, 1) == size(B, 1)
    n = size(A, 2)
    basis = nullspace([A -B])
    basis[1:n, :], basis[n + 1:end, :]
end

# Compute the pushout of the diagram
#  l <- A - m - B -> n
# in the category of matrices.
function pushout(A::AbstractMatrix, B::AbstractMatrix)
    map(transpose, pullback(A', B'))
end

# Compute the universal morphism into the pullback of the diagram
#  l - A -> m <- B - n
# in category of matrices.
function universal_pullback(
    A::AbstractMatrix,
    B::AbstractMatrix,
    C::AbstractMatrix,
    D::AbstractMatrix)

U, V = map(adjoint, pullback(A, B))
U * C + V * D
end



# Compute the universal morphism out of pushout of the diagram
#  l <- A - m - B -> n
# in category of matrices.
function universal_pushout(
        A::AbstractMatrix,
        B::AbstractMatrix,
        C::AbstractMatrix,
        D::AbstractMatrix)

    universal_pullback(A', B', C', D')
end

function product(n::Int, m::Int)
    p1 = [I(n) zeros(n,m)]
    p2 = [zeros(m,n) I(m)]
    return p1,p2
end

pair(A::AbstractMatrix, B::AbstractMatrix) = [A; B]


# Tests

A = [0.0 1.0]
B = [1.0 0.0]

C,D = pullback(A,B)

x = rand(3)

@test norm(A*C*x - B*D*x) < 1e-12

um = [0.0 0.0]

π1,π2 = pullback(um,um)

ϕ = universal_pullback(um,um,C,D)

struct OpenObjective # X -> Y
    obj::Function # V -> R
    A::AbstractMatrix # V -> X
    B::AbstractMatrix # V -> Y
    # TODO: Add test that dom(A) == dom(B)
end

(f::OpenObjective)(x) = f.obj(x)
dom(f::OpenObjective) = size(f.A, 1)
codom(f::OpenObjective) = size(f.B, 1)
apex(f::OpenObjective) = size(f.A, 2)

function compose(f::OpenObjective, g::OpenObjective)
    C,D = pullback(f.B, g.A)
    ϕ = pair(C,D)

    X = apex(f)
    Y = apex(g)
    p1, p2 = product(X,Y)

    comp_obj(x) = f.obj(p1*ϕ*x) + g.obj(p2*ϕ*x)
    return OpenObjective(comp_obj, f.A*C, g.B*D)
end

Q = rand(4,4)
Q = Q'*Q

R = rand(5,5)
R = R'*R

A1 = rand(3,4)
B1 = rand(2,4)

A2 = rand(2,5)
B2 = rand(3, 5)

f = OpenObjective(x -> x'*Q*x, A1, B1)
g = OpenObjective(x -> x'*R*x, A2, B2)

gf = compose(f, g)




