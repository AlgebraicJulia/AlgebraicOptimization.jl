module BlockSparseArrays

using ArgCheck
using Base: oneto
using LinearAlgebra
using SparseArrays

export BlockSparseMatrixCSC, blocksparse

function half(i::I) where {I}
    return div(i, convert(I, 2))
end

function ispositive(i::I) where {I}
    return i > zero(I)
end

struct BlockSparseMatrixCSC{T, I} <: AbstractMatrix{T}
    nrow::I
    ncol::I
    nadj::I
    nrowblk::I
    ncolblk::I
    nadjblk::I
    xadj::Vector{I}
    xrowblk::Vector{I}
    xcolblk::Vector{I}
    xadjblk::Vector{I}
    adj::Vector{I}
    adjblk::Vector{T}

    function BlockSparseMatrixCSC{T, I}(
            nrow::Integer, 
            ncol::Integer,
            nadj::Integer,
            nrowblk::Integer,
            ncolblk::Integer,
            nadjblk::Integer,
            xadj::AbstractVector,
            xrowblk::AbstractVector,
            xcolblk::AbstractVector,
            xadjblk::AbstractVector,
            adj::AbstractVector,
            adjblk::AbstractVector,
        ) where {T, I}
        @argcheck 0 <= nrow
        @argcheck 0 <= ncol
        @argcheck 0 <= nadj
        @argcheck 0 <= nrowblk
        @argcheck 0 <= ncolblk
        @argcheck 0 <= nadjblk
        @argcheck ncol < length(xadj)
        @argcheck ncol < length(xadjblk)
        @argcheck nadj <= length(adj)
        @argcheck nadj <= length(adjblk)
        
        return new{T, I}(nrow, ncol, nadj, nrowblk, ncolblk,
            nadjblk, xadj, xrowblk, xcolblk, xadjblk, adj, adjblk)
    end
end

function SparseArrays.sparse(A::BlockSparseMatrixCSC{T, I}) where {T, I}
    m = A.nrowblk
    n = A.ncolblk
    nnz = A.nadjblk
    
    colptr = Vector{I}(undef, n + one(I))
    rowval = Vector{I}(undef, nnz)
    nzval = Vector{T}(undef, nnz)

    colptr[one(I)] = pblk = one(I)

    for j in oneto(A.ncol)
        colblkstrt = A.xcolblk[j]
        colblkstop = A.xcolblk[j + one(I)] - one(I)
        
        adjstrt = A.xadj[j]
        adjstop = A.xadj[j + one(I)] - one(I)

        for jblk in colblkstrt:colblkstop
            for p in adjstrt:adjstop
                i = A.adj[p]
    
                rowblkstrt = A.xrowblk[i]
                rowblkstop = A.xrowblk[i + one(I)] - one(I)
                    
                adjblkstrt = A.xadjblk[p]
                adjblkstop = A.xadjblk[p + one(I)] - one(I)

                Ap = reshape(
                    view(A.adjblk, adjblkstrt:adjblkstop),
                    rowblkstop - rowblkstrt + one(I),
                    colblkstop - colblkstrt + one(I),
                )
                
                for iblk in rowblkstrt:rowblkstop
                    ip = iblk - rowblkstrt + one(I)
                    jp = jblk - colblkstrt + one(I)

                    rowval[pblk] = iblk
                    nzval[pblk] = Ap[ip, jp]
                    pblk += one(I)
                end
            end

            colptr[jblk + one(I)] = pblk
        end
    end

    return SparseMatrixCSC{T, I}(m, n, colptr, rowval, nzval)
end

function blocksparse(I, J, V, args...)
    matrix = sparse(I, J, eachindex(V), args...)
    return blocksparse(matrix, V)
end

function blocksparse(matrix::SparseMatrixCSC{<:Any, I}, V::AbstractVector{<:AbstractMatrix{T}}) where {T, I}
    nrow = convert(I, size(matrix, 1))
    ncol = convert(I, size(matrix, 2))
    nadj = convert(I, nnz(matrix))

    nrowblk = zero(I)
    ncolblk = zero(I)
    nadjblk = zero(I)
    
    xrowblk = Vector{I}(undef, nrow + one(I))
    xcolblk = Vector{I}(undef, ncol + one(I))
    xadjblk = Vector{I}(undef, nadj + one(I))

    xrowblk[one(I)] = nrowblk + one(I)
    xcolblk[one(I)] = ncolblk + one(I)
    xadjblk[one(I)] = nadjblk + one(I)

    for i in oneto(nrow)
        xrowblk[i + one(I)] = zero(I)
    end

    for i in oneto(ncol)
        xcolblk[i + one(I)] = zero(I)
    end
    
    xadj = matrix.colptr
    adj = matrix.rowval
    val = matrix.nzval

    for p in oneto(nadj)
        A = V[val[p]]
        nadjblk += convert(I, size(A, 1) * size(A, 2))
        xadjblk[p + one(I)] = nadjblk + one(I)
    end

    adjblk = Vector{T}(undef, nadjblk)
    
    for j in oneto(ncol)
        adjstrt = xadj[j]
        adjstop = xadj[j + one(I)] - one(I)

        for p in adjstrt:adjstop
            i = adj[p]; A = V[val[p]]

            xrowblk[i + one(I)] = rowblkdeg = convert(I, size(A, 1))
            xcolblk[j + one(I)] = colblkdeg = convert(I, size(A, 2))

            adjblkstrt = xadjblk[p]
            adjblkstop = xadjblk[p + one(I)] - one(I)

            B = reshape(
                view(adjblk, adjblkstrt:adjblkstop),
                rowblkdeg,
                colblkdeg,
            )

            copyto!(B, A)
        end
    end

    for i in oneto(nrow)
        nrowblk += xrowblk[i + one(I)]
        xrowblk[i + one(I)] = nrowblk + one(I)
    end

    for i in oneto(ncol)
        ncolblk += xcolblk[i + one(I)]
        xcolblk[i + one(I)] = ncolblk + one(I)
    end

    return BlockSparseMatrixCSC{T, I}(nrow, ncol, nadj, nrowblk,
        ncolblk, nadjblk, xadj, xrowblk, xcolblk, xadjblk, adj, adjblk)
end

function bspgetidx(A::BlockSparseMatrixCSC{T, I}, blki::I, blkj::I) where {T, I}
    @boundscheck checkbounds(axes(A, 1), blki)
    @boundscheck checkbounds(axes(A, 2), blkj)

    Aij = zero(T)

    jstrt = one(I)
    jstop = A.ncol

    while jstrt <= jstop
        jcent = jstrt + half(jstop - jstrt)

        if A.xcolblk[jcent] <= blkj
            jstrt = jcent + one(I)
        else
            jstop = jcent - one(I)
        end
    end

    j = jstrt - one(I)
    
    colblkstrt = A.xcolblk[j]
    colblkstop = A.xcolblk[j + one(I)] - one(I)
    
    adjstrt = A.xadj[j]
    adjstop = A.xadj[j + one(I)] - one(I)

    pstrt = adjstrt
    pstop = adjstop

    while pstrt <= pstop
        pcent = pstrt + half(pstop - pstrt)

        if A.xrowblk[A.adj[pcent]] <= blki
            pstrt = pcent + one(I)
        else
            pstop = pcent - one(I)
        end
    end

    p = pstrt - one(I)
    
    if p >= adjstrt
        i = A.adj[p]
        
        rowblkstrt = A.xrowblk[i]
        rowblkstop = A.xrowblk[i + one(I)] - one(I)
    
        if blki <= rowblkstop
            adjblkstrt = A.xadjblk[p]
            adjblkstop = A.xadjblk[p + one(I)] - one(I)
    
            ip = blki - rowblkstrt + one(I)
            jp = blkj - colblkstrt + one(I)
        
            Ap = reshape(
                view(A.adjblk, adjblkstrt:adjblkstop),
                rowblkstop - rowblkstrt + one(I),
                colblkstop - colblkstrt + one(I),
            )
    
            Aij = Ap[ip, jp]
        end
    end
    
    return Aij
end

function bspvecmul_N!(
        C::AbstractVector,
        A::BlockSparseMatrixCSC{<:Any, I},
        B::AbstractVector,
        α::Number,
        β::Number,
    ) where {I}    
    @argcheck size(A, 1) == length(C)
    @argcheck size(A, 2) == length(B)

    C .*= β

    for j in oneto(A.ncol)
        colblkstrt = A.xcolblk[j]
        colblkstop = A.xcolblk[j + one(I)] - one(I)
        
        Bj = view(B, colblkstrt:colblkstop)

        adjstrt = A.xadj[j]
        adjstop = A.xadj[j + one(I)] - one(I)
        
        for p in adjstrt:adjstop
            i = A.adj[p]

            rowblkstrt = A.xrowblk[i]
            rowblkstop = A.xrowblk[i + one(I)] - one(I)
            
            Ci = view(C, rowblkstrt:rowblkstop)

            adjblkstrt = A.xadjblk[p]
            adjblkstop = A.xadjblk[p + one(I)] - one(I)

            Ap = reshape(
                view(A.adjblk, adjblkstrt:adjblkstop),
                rowblkstop - rowblkstrt + one(I),
                colblkstop - colblkstrt + one(I),
            )

            mul!(Ci, Ap, Bj, α, true)
        end
    end

    return C
end

function bspvecmul_TC!(
        f::F,
        C::AbstractVector,
        A::BlockSparseMatrixCSC{<:Any, I},
        B::AbstractVector,
        α::Number,
        β::Number,
    ) where {F, I}
    @argcheck size(A, 2) == length(C)
    @argcheck size(A, 1) == length(B)

    C .*= β

    for j in oneto(A.ncol)
        colblkstrt = A.xcolblk[j]
        colblkstop = A.xcolblk[j + one(I)] - one(I)
        
        Cj = view(C, colblkstrt:colblkstop)

        adjstrt = A.xadj[j]
        adjstop = A.xadj[j + one(I)] - one(I)
        
        for p in adjstrt:adjstop
            i = A.adj[p]

            rowblkstrt = A.xrowblk[i]
            rowblkstop = A.xrowblk[i + one(I)] - one(I)
            
            Bi = view(B, rowblkstrt:rowblkstop)

            adjblkstrt = A.xadjblk[p]
            adjblkstop = A.xadjblk[p + one(I)] - one(I)

            Ap = reshape(
                view(A.adjblk, adjblkstrt:adjblkstop),
                rowblkstop - rowblkstrt + one(I),
                colblkstop - colblkstrt + one(I),
            )

            mul!(Cj, Ap |> f, Bi, α, true)
        end
    end

    return C
end

# ======================== #
# Abstract Array Interface #
# ======================== #

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::BlockSparseMatrixCSC,
        B::AbstractVector,
        α::Number,
        β::Number,
    )
    bspvecmul_N!(C, A, B, α, β)
    return C
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::Transpose{<:Any, <:BlockSparseMatrixCSC},
        B::AbstractVector,
        α::Number,
        β::Number,
    )
    bspvecmul_TC!(transpose, C, parent(A), B, α, β)
    return C
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::Adjoint{<:Any, <:BlockSparseMatrixCSC},
        B::AbstractVector,
        α::Number,
        β::Number,
    )
    bspvecmul_TC!(adjoint, C, parent(A), B, α, β)
    return C
end

function Base.getindex(A::BlockSparseMatrixCSC{<:Any, I}, i::Integer, j::Integer) where {I}
    return bspgetidx(A, convert(I, i), convert(I, j))
end

function Base.IndexStyle(::Type{<:BlockSparseMatrixCSC})
    return IndexCartesian()
end

function Base.size(A::BlockSparseMatrixCSC)
    m = convert(Int, A.nrowblk)
    n = convert(Int, A.ncolblk)
    return (m, n)
end

end
