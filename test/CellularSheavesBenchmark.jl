using AlgebraicOptimization
using Test
using BenchmarkTools


# New test cases: ThreadedSheafs, shared memory

sheaf_2 = ThreadedSheaf([2, 2], [1])
add_map!(sheaf_2, 1, 1, [0 1])
add_map!(sheaf_2, 2, 1, [0 1]) 

sheaf_2.f = [x -> x[1]^2 + x[2]^2, x -> (x[1] - 2)^2 + (x[2] -2)^2]
@btime simulate!(sheaf_2)

# ThreadedSheaf, sequential version (should be slower, hopefully)

sheaf_2 = ThreadedSheaf([2, 2], [1])
add_map!(sheaf_2, 1, 1, [0 1])
add_map!(sheaf_2, 2, 1, [0 1]) 

sheaf_2.f = [x -> x[1]^2 + x[2]^2, x -> (x[1] - 2)^2 + (x[2] -2)^2]
@btime simulate_sequential!(sheaf_2)













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