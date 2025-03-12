using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays


c = CellularSheaf([4, 4, 4], [4, 4, 4])

set_edge_maps!(c, 1, 2, 1, I(4), I(4))
set_edge_maps!(c, 1, 3, 2, I(4), I(4))
set_edge_maps!(c, 2, 3, 3, I(4), I(4))

x = BlockArray(rand(12), c.vertex_stalks)

y = nearest_section(c, x)

average_x = (x[Block(1)] + x[Block(2)] + x[Block(3)]) / 3

@test y[Block(1)] - average_x ≈ zeros(4) atol = 1e-5
