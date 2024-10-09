using AlgebraicOptimization
using Test


# Example sheaf 1: (a, b) -> b <- (b, c)
# Underlying graph: *-------*  (2 vertices, 1 edge)
# a, b, and c are each 1-element vectors
# Restriction maps go from b to b and all other elements to 0

sheaf_1 = CellularSheaf(2, 1, 2)

add_map!(sheaf_1, 1, 1, [0. 1.; 0. 0.]);
add_map!(sheaf_1, 2, 1, [-1. 0.; 0. 0.])  # Note: Right now I'm manually negating the linear maps so that they work in the coboundary map. My next step
                                          # is to implement BlockArrays so that we can do automatic negating easily.
println(sheaf_1.coboundary_map)

test_arr = [1 2 3 4]   # V1 holds (1, 2) and V2 holds (3, 4)
@test !(sheaf_1.coboundary_map * test_arr' ≈ [0, 0])

test_arr = [1 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test sheaf_1.coboundary_map * test_arr' ≈ [0, 0]

test_arr = [15 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test sheaf_1.coboundary_map * test_arr' ≈ [0, 0]