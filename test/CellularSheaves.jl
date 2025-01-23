using AlgebraicOptimization
using Test
using Graphs



# Example sheaf 1: (a, b) -> b <- (b, c)
# Underlying graph: *-------*  (2 vertices, 1 edge)
# a, b, and c are each 1-element vectors
# Restriction maps go from b to b and all other elements to 0

sheaf_1 = CellularSheaf(2, 1, 2)
add_map!(sheaf_1, 1, 1, [0 1; 0 0])
add_map!(sheaf_1, 2, 1, [1 0; 0 0]) 

# Tests
test_arr = [1 2 3 4]   # V1 holds (1, 2) and V2 holds (3, 4)     Could switch to a list w/ commas (vector) to avoid '
@test !(laplacian(sheaf_1) * test_arr' ≈ [0, 0, 0, 0])

test_arr = [1 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test laplacian(sheaf_1) * test_arr' ≈ [0, 0, 0, 0]

test_arr = [15 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test laplacian(sheaf_1) * test_arr' ≈ [0, 0, 0, 0]



# Example sheaf 2: (a, b) -> b <- (b, c)
# This is the same as example 1 except the vector space on the b edge has dimension 1, not 2
# i.e. not all dimensions are the same (this is a more lightweight approach because we don't give the b edge an "extra" throwaway dimension)
# Restriction maps go from b to b and all other elements to 0

sheaf_2 = CellularSheaf([2, 2], [1])
add_map!(sheaf_2, 1, 1, [0 1])
add_map!(sheaf_2, 2, 1, [1 0]) 

# Tests
test_arr = [1 2 3 4]   # V1 holds (1, 2) and V2 holds (3, 4)
@test !(laplacian(sheaf_2) * test_arr' ≈ [0, 0, 0, 0])

test_arr = [1 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test laplacian(sheaf_2) * test_arr' ≈ [0, 0, 0, 0]

test_arr = [15 2 2 4]   # V1 holds (1, 2) and V2 holds (2, 4)
@test laplacian(sheaf_2) * test_arr' ≈ [0, 0, 0, 0]



# Example sheaf 3: (a, b) -> b <- (b, c)
# Underlying graph: *-------*  (2 vertices, 1 edge)
# a, b, and c are each 2-element vectors (unlike example sheaf 1)
# Restriction maps go from b to b and all other elements to 0

sheaf_3 = CellularSheaf(2, 1, 4)
add_map!(sheaf_3, 1, 1, [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0])
add_map!(sheaf_3, 2, 1, [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]) 

# Tests
test_arr = [1 2 3 4 5 6 7 8]   # V1 holds (a =(1, 2), b =(3, 4)). V2 holds (b =(5, 6), c =(7, 8))
@test !(laplacian(sheaf_3) * test_arr' ≈ zeros(8))

test_arr = [1 2 3 4 3 4 7 8]   # V1 holds (a =(1, 2), b =(3, 4)). V2 holds (b =(3, 4), c =(7, 8))
@test laplacian(sheaf_3) * test_arr' ≈ zeros(8)

test_arr = [1 2 3 4 3 100 7 8]   # V1 holds (a =(1, 2), b =(3, 4)). V2 holds (b =(3, 100), c =(7, 8))
@test !(laplacian(sheaf_3) * test_arr' ≈ zeros(8))



# Example sheaf 4: Vertices are (a, b), (b, c), and (c, a). Edges connect the shared variables as in examples 1 and 2.
# Underlying graph is a triangle
# a, b, and c are each 1-element vectors

sheaf_4 = CellularSheaf(3, 3, 2)

add_map!(sheaf_4, 1, 1, [0 1; 0 0])
add_map!(sheaf_4, 2, 1, [1 0; 0 0]) 

add_map!(sheaf_4, 2, 2, [0 1; 0 0])
add_map!(sheaf_4, 3, 2, [1 0; 0 0]) 

add_map!(sheaf_4, 3, 3, [0 1; 0 0])
add_map!(sheaf_4, 1, 3, [1 0; 0 0]) 

# Tests
test_arr = [10 20 20 30 30 50] # V1 holds (1, 2), V2 holds (20, 30), V3 holds (30, 50)
@test !(laplacian(sheaf_4) * test_arr' ≈ zeros(6))

test_arr = [10 20 20 30 30 10]   # V1 holds (1, 2), V2 holds (20, 30), V3 holds (30, 10)
@test laplacian(sheaf_4) * test_arr' ≈ zeros(6)


# SheafObjective test: Uzawa's algorithm. Currently not very distributed.
# Not sure why but currently these are failing

# sheaf_5 = CellularSheaf(2, 1, 1)
# add_map!(sheaf_5, 1, 1, reshape([10], 1, 1))
# add_map!(sheaf_5, 2, 1, reshape([10], 1, 1))

# sheaf_objective_5 = SheafObjective([x -> x^2, x -> (x -2)^2], sheaf_5, [4, 6], [0, 0])

# apply_f_with_stabilizer(sheaf_objective_5)

# simulate!(sheaf_objective_5)

# @test sheaf_objective_5.x ≈ [10, 10] atol=1e-2
# sheaf_objective_5.z





# Distributed solvers using SheafNodes
# Similar test problem to the previous test case using SheafObjectives

sheaf_node_1 = SheafNode( x -> (x[2])^2, [1, 3], [2, 4])
sheaf_node_2 = SheafNode(x -> (x[1] - 2)^2, [22, 7], [8, 6])
add_edge!(sheaf_node_1, sheaf_node_2, [0 1], [1 0])


distributed_sheaf_1 = [sheaf_node_1, sheaf_node_2]

simulate_distributed!(distributed_sheaf_1, .1, 100)
# simulate!(distributed_sheaf_1, .1, 100)

@test distributed_sheaf_1[1].x[2] ≈ 1  atol=1e-3
@test distributed_sheaf_1[2].x[1] ≈ 1  atol=1e-3


# SheafNode: more involved example
#    (a, b)       (b, c)

#          (c, a)

sheaf_node_3 = SheafNode( x -> (x[1]^2 + x[2]^2), [1, 3], [0, 0])
sheaf_node_4 = SheafNode(x -> x[1]^2 + x[2]^2, [22, 7], [0, 0])
sheaf_node_5 = SheafNode(x -> (x[1] + x[2])^2, [22, 7], [0, 0])

add_edge!(sheaf_node_3, sheaf_node_4, [1 0], [0 1])
add_edge!(sheaf_node_4, sheaf_node_5, [1 0], [0 1])
add_edge!(sheaf_node_5, sheaf_node_3, [1 0], [0 1])


distributed_sheaf_2 = [sheaf_node_3, sheaf_node_4, sheaf_node_5]    # Should this be a shared array? Would that change anything?
distributed_sheaf_2_copy = deepcopy(distributed_sheaf_2)

simulate_distributed!(distributed_sheaf_2, .1, 1000)
# simulate!(distributed_sheaf_2_copy, .1, 1000)

@test distributed_sheaf_2[1].x ≈ [0, 0]  atol=1e-3
@test distributed_sheaf_2[2].x ≈ [0, 0]  atol=1e-3
@test distributed_sheaf_2[3].x ≈ [0, 0]  atol=1e-3


# Constant sheaf

sheaf_node_6 = SheafNode(x -> x[1]^2 + (x[2] - 3)^2, [3, 4], [0, 0])
sheaf_node_7 = SheafNode(x -> x[1]^2 + (x[2] - 3)^2, [-2, 5], [0, 0])   
add_edge!(sheaf_node_6, sheaf_node_7, [1 0; 0 1], [1 0; 0 1])   # Identity matrix since this is the constant sheaf

distributed_sheaf_3 = [sheaf_node_6, sheaf_node_7]
simulate_distributed_separate_steps!(distributed_sheaf_3, .01, 1000)

@test sheaf_node_6.x[1] ≈ 0  atol=1e-3            # TODO: These are the bad tests
@test sheaf_node_6.x[2] ≈ 3  atol=1e-3


# Constant sheaf: SheafVertex instead of SheafNode (Dr. Fairbanks' approach)

sheaf_vertex_6 = SheafVertex(x -> x[1]^2, [3, 4], [0, 0])
sheaf_vertex_7 = SheafVertex(x -> (x[2] - 3)^2, [-2, 5], [0, 0])   
add_edge!(sheaf_vertex_6, sheaf_vertex_7, [1 0; 0 1], [1 0; 0 1])   # Identity matrix since this is the common sheaf

distributed_sheaf_4 = [sheaf_vertex_6, sheaf_vertex_7]
simulate_distributed!(distributed_sheaf_4, .1, 100)

@test sheaf_vertex_6.x[1] ≈ 0  atol=1e-3
@test sheaf_vertex_6.x[2] ≈ 3  atol=1e-3



# New test cases: MatrixSheafs, shared memory

sheaf_2 = MatrixSheaf([2, 2], [1])
add_map!(sheaf_2, 1, 1, [0 1])
add_map!(sheaf_2, 2, 1, [0 1]) 

sheaf_2.f = [x -> x[1]^2 + x[2]^2, x -> (x[1] - 2)^2 + (x[2] -2)^2]
simulate!(sheaf_2)

@test sheaf_2.x ≈ [0; 1; 2; 1]  atol=1e-3  


# Random constructor for MatrixSheaf that takes in a Graph

V = 10
E = 15
dim = 3
sparsity = 1.0
er = erdos_renyi(V, E)

for i in edges(er)
    println(i)
end

test_graph_sheaf = random_matrix_sheaf(er, dim, sparsity)

# Number of non-zero maps in the restriction map matrix should equal twice the total number of edges in the graph

num_restriction_maps = 0
for v in 1:V
    for e in 1:E
        if !iszero(test_graph_sheaf.restriction_maps[Block(e, v)])
            num_restriction_maps += 1
        end
    end
end

@test num_restriction_maps == 2 * E

# Next steps: 10/28/24

# Add more test cases for the newer approach with separate vertices and edges

# Ask in lab meeting for how to make it more distributed (MPI, partitioning/clustering, etc.)

# Test on a big enough problem so that @distributed actually has speedup

# Random quadratic forms on all the vertices
# Random graph with some specific properties?

# Do we need the ability to read in a graph?
# Random positive definite matrix (P^T P)

# x^T Q x + b^T x        <--- objective functions         # + c
# Restriction maps: arbitrary full rank square matrices    <---- no consensus at all?   (not true--there's always 0)
# Line graph, mpc example?  (too simple)

# Identities (constant sheaf) is always a good baseline.
# 0/1 projection matrices.