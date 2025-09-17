using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables
include("PaperPlotting.jl")
using .PaperPlotting


# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0]
system = DiscreteLinearSystem(A, B, C)

# TEST CASE 1: Consensus, with x unconstrained and y constrained.

# Set up each agent's objective function: x'Qx + u'Ru
Q = I(4)
Q[1, 1] = 0    # First variable is unconstrained
R = I(2)


# Set up system properties: time horoizon and control bounds
N = 10
control_bounds = [-2.0, 2.0]
params1 = MPCParams(Q, R, system, control_bounds, N, [0, 0, 2.0, 0])
params2 = MPCParams(Q, R, system, control_bounds, N, [0, 0, 4.0, 0])
params3 = MPCParams(Q, R, system, control_bounds, N, [0, 0, 6.0, 0])


# Set up communication pattern: triangular sheaf
c = CellularSheaf([4, 4, 4], [1, 1, 1])
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)


# Set up solver
x_init = BlockArray(5 * rand(-1:0.1:1, 12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params1, params2, params3], c, x_init)
alg = ADMM(2.0, 10)
num_iters = 100


# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)


# Plot results
PaperPlotting.paper_plot_save_results(trajectory, [1 0 0 0; 0 0 1 0], "Consensus", 2, "x unconstrained and y constrained")