using AlgebraicOptimization
using Test
using BenchmarkTools
using LinearAlgebra
using Random

Random.seed!(3)


# println("Number of threads: ", Threads.nthreads())   # 20 on Sam's laptop
BLAS.set_num_threads(20)   # Why did this slow me down?


# # ThreadedSheaf, sequential version (should be slower, hopefully)

# sheaf_2 = ThreadedSheaf([2, 2], [1])

# add_map!(sheaf_2, 1, 1, [0 1])
# add_map!(sheaf_2, 2, 1, [0 1]) 
# sheaf_2.f = [x -> x[1]^2 + x[2]^2, x -> (x[1] - 2)^2 + (x[2] -2)^2]

# sheaf_2_sequential = deepcopy(sheaf_2)

# println("sheaf_2 parallel version:")
# @btime simulate!(sheaf_2)   # Would @time be better since this method modifies? Maybe we want a version that doesn't modify for benchmarking purposes?

# println("sheaf_2 sequential version:")
# @btime simulate_sequential!(sheaf_2_sequential)



# Big sheaf 2: V many vertices, E many edges
# For simplicity of constructing, we're going to allow parallel edges

V = 10
E = 10
dim = 300

big_sheaf_2 = ThreadedSheaf([dim for _ in 1:V], [dim for _ in 1:E])  # Could we speed this up?

# Add random restriction maps
for e in 1:E
    V = 10  # Example value
    u = rand(1:V)
    w = rand(1:V)
    while w == u
        w = rand(1:V)  # Keep generating until b is different from a
    end
    add_map!(big_sheaf_2, u, e, rand(dim, dim))
    add_map!(big_sheaf_2, w, e, rand(dim, dim))
end

# Add random objective functions
big_sheaf_2.f = [let Q = rand(dim, dim), b = rand(1, dim) 
                     x -> only(x' * Q * x + b * x) 
                 end for _ in 1:V]

big_sheaf_2_sequential = deepcopy(big_sheaf_2)



simulate_sequential!(big_sheaf_2_sequential, 1e-4, 1)
simulate!(big_sheaf_2, 1e-4, 1)


println("big_sheaf_2 sequential version:")
@time simulate_sequential!(big_sheaf_2_sequential, 1e-4, 5)

println("big_sheaf_2 parallel version:")
@time simulate!(big_sheaf_2, 1e-4, 5)   # Add on checks that big_sheaf has all the right dimensions, etc?   -4 seems to be a sweet spot...





#  Run command:   julia --threads=auto --project=. C:\Users\samco\OneDrive\Desktop\AlgOptOfficial\AlgebraicOptimization.jl\test\CellularSheavesBenchmark.jl


# Why is @threads slower? Can we make it faster?
# How to well condition the problem so we reach a solution?
# Incorporate Octavian to speed up the matrix multiply





# Basic thread testing

# random = zeros(100)

# function fill(random)
#     Threads.@threads for i in eachindex(random)
#         random[i] = rand()  # Assign a random float between 0 and 1
#     end
# end


# function fill_sequential(random)
#     for i in eachindex(random)
#         random[i] = rand()  # Assign a random float between 0 and 1
#     end
# end

# @btime fill(random)
# println(random)
# println("num threads: ", Threads.nthreads())

# @btime fill_sequential(random)

# @btime fill(random)