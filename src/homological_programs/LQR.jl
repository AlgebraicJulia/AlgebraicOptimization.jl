# Multi-agent LQR problems defined by coordination sheaves.

using MatrixEquations
using AlgebraicOptimization
using LinearAlgebra
using SparseArrays

C = [1.0 0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 0]


s = @cellular_sheaf C begin
    x::Stalk{4}, y::Stalk{4}, z::Stalk{4}

    C(x) == C(y)
    C(y) == C(z)
end

#Q = sheaf_laplacian_matrix(s) + diagm(repeat([1.0], 12))
Q = I(12)
R = I(6)

A_i = [0 1.0 0 0; 0 0 0 0; 0 0 1 0; 0 0 0 0]
B_i = [0 0; 1.0 0; 0 0; 0 1]

A = Array(sparse(blocksparse([1, 2, 3], [1, 2, 3], [A_i, A_i, A_i])))
B = Array(sparse(blocksparse([1, 2, 3], [1, 2, 3], [B_i, B_i, B_i])))

X = arec(A, B, R, Q)
