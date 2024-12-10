#  Run command:   julia --threads=auto --project=. C:\Users\samco\OneDrive\Desktop\AlgOptOfficial\AlgebraicOptimization.jl\test\CellularSheavesBenchmark.jl
#  REPL startup command: julia --threads=auto

using AlgebraicOptimization
using Test
using BenchmarkTools
using LinearAlgebra
using Random
using CairoMakie

Random.seed!(3)
BLAS.set_num_threads(1)


# Runtime vs. Stalk Dimension for ThreadedSheaves

V = 20          # Number of vertices
E = 20          # Number of edges
dims = [1, 5, 10, 25, 50, 100, 200, 300, 400, 500]  # Range of dimensions to test
times_simulate = Float64[]
times_simulate_sequential = Float64[]

for dim in dims
    println("\nBenchmarking for dim = ", dim)

    # Initialize big_sheaf_2 for the current dim
    big_sheaf_2 = random_threaded_sheaf(V, E, dim)
    big_sheaf_2_sequential = deepcopy(big_sheaf_2)

    # Warm-up to avoid precompilation time
    println("Warming up simulate_sequential! and simulate!")
    simulate_sequential!(big_sheaf_2_sequential, 1e-4, 1)
    simulate!(big_sheaf_2, 1e-4, 1)

    # Measure time for simulate_sequential!
    time_seq = @elapsed simulate_sequential!(big_sheaf_2_sequential, 1e-4, 5)
    println("Sequential time for dim = $dim: $time_seq seconds")

    # Measure time for simulate!
    time_par = @elapsed simulate!(big_sheaf_2, 1e-4, 5)
    println("Parallel time for dim = $dim: $time_par seconds")

    # Store results
    push!(times_simulate_sequential, time_seq)
    push!(times_simulate, time_par)
end


# Graph results
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1],
    title = "Runtime vs. Stalk Dimension for ThreadedSheaves",
    xlabel = "Dimension",
    ylabel = "Runtime (seconds)"
)
lines!(ax, dims, times_simulate_sequential; label="simulate_sequential!", linewidth=2, color=:blue)
lines!(ax, dims, times_simulate; label="simulate!", linewidth=2, color=:red)
save("performance_comparison_more_dims.png", fig)


# Runtime vs. Number of BLAS threads

V = 20          # Number of vertices
E = 20          # Number of edges
dim = 300  # Range of dimensions to test
thread_counts = [i for i in 1:20]
times_blas = Float64[]


big_sheaf_2 = random_threaded_sheaf(V, E, dim)
for blas in thread_counts
    println("\nBenchmarking for blas = ", blas)
    BLAS.set_num_threads(blas)

    # Initialize big_sheaf_2 for the current dim
    big_sheaf_2_copy = deepcopy(big_sheaf_2)
    big_sheaf_2_copy_2 = deepcopy(big_sheaf_2)

    # Warm-up to avoid precompilation time
    println("Warming up simulate!")
    simulate!(big_sheaf_2_copy, 1e-4, 1)

    # Measure time for simulate!
    time_par = @belapsed simulate!($big_sheaf_2_copy_2, 1e-4, 5)
    println("Parallel time for blas = $blas: $time_par seconds")

    # Store results
    push!(times_blas, time_par)
end

# Make graph
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1],
    title = "Runtime vs. Number of BLAS threads",
    xlabel = "Number of BLAS threads",
    ylabel = "Runtime (seconds)"
)
ylims!(ax, 0, 1) 
lines!(ax, thread_counts, times_blas; linewidth=2, color=:blue)
save("blas_comparison_belapsed.png", fig)


# Why is @threads slower? Can we make it faster?]
# How to well condition the problem so we reach a solution?
# Incorporate Octavian to speed up the matrix multiply