using AlgebraicOptimization
using Test


# Example sheaf 1: (a, b) -> b <- (b, c)
# Underlying graph: *-------*  (2 vertices, 1 edge)
# a, b, and c are each 1-element vectors
# Restriction maps go from b to b and all other elements to 0

sheaf_1 = CellularSheaf(2, 1, 2)
add_map!(sheaf_1, 1, 1, [0 1.; 0 0])
add_map!(sheaf_1, 2, 1, [1. 0; 0 0]) 

# Tests
test_arr = [1 2 3 4]   # V1 holds (1, 2) and V2 holds (3, 4)
@test !(laplacian(sheaf_1) * test_arr' ≈ [0, 0, 0, 0])

test_arr = [1 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test laplacian(sheaf_1) * test_arr' ≈ [0, 0, 0, 0]

test_arr = [15 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test laplacian(sheaf_1) * test_arr' ≈ [0, 0, 0, 0]



# Example sheaf 2: (a, b) -> b <- (b, c)
# Underlying graph: *-------*  (2 vertices, 1 edge)
# a, b, and c are each 2-element vectors (unlike example sheaf 1)
# Restriction maps go from b to b and all other elements to 0

sheaf_2 = CellularSheaf(2, 1, 4)
add_map!(sheaf_2, 1, 1, [0 0 1. 0; 0 0 0 1; 0 0 0 0; 0 0 0 0])
add_map!(sheaf_2, 2, 1, [1. 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]) 

# Tests
test_arr = [1 2 3 4 5 6 7 8]   # V1 holds (a =(1, 2), b =(3, 4)). V2 holds (b =(5, 6), c =(7, 8))
@test !(laplacian(sheaf_2) * test_arr' ≈ zeros(8))

test_arr = [1 2 3 4 3 4 7 8]   # V1 holds (a =(1, 2), b =(3, 4)). V2 holds (b =(3, 4), c =(7, 8))
@test laplacian(sheaf_2) * test_arr' ≈ zeros(8)

test_arr = [1 2 3 4 3 100 7 8]   # V1 holds (a =(1, 2), b =(3, 4)). V2 holds (b =(3, 100), c =(7, 8))
@test !(laplacian(sheaf_2) * test_arr' ≈ zeros(8))


# Example sheaf 3: Vertices are (a, b), (b, c), and (c, a). Edges connect the shared variables as in examples 1 and 2.
# Underlying graph is a triangle
# a, b, and c are each 1-element vectors

sheaf_3 = CellularSheaf(3, 3, 2)

add_map!(sheaf_3, 1, 1, [0 1.; 0 0])
add_map!(sheaf_3, 2, 1, [1. 0; 0 0]) 

add_map!(sheaf_3, 2, 2, [0 1.; 0 0])
add_map!(sheaf_3, 3, 2, [1. 0; 0 0]) 

add_map!(sheaf_3, 3, 3, [0 1.; 0 0])
add_map!(sheaf_3, 1, 3, [1. 0; 0 0]) 

# Tests
test_arr = [10 20 20 30 30 50] # V1 holds (1, 2), V2 holds (20, 30), V3 holds (30, 50)
@test !(laplacian(sheaf_3) * test_arr' ≈ zeros(6))

test_arr = [10 20 20 30 30 10]   # V1 holds (1, 2), V2 holds (20, 30), V3 holds (30, 10)
@test laplacian(sheaf_3) * test_arr' ≈ zeros(6)
