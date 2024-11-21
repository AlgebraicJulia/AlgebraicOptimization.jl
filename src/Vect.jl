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
end

function compose(f::OpenObjective, g::OpenObjective)
    C,D = pullback(f.B, g.A)
    ϕ = [C;D]
    

end


