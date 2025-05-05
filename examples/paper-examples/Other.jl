using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables
using .PaperPlotting

# TEST CASE 1: Formation, unconstrained, on the curve xy = 1

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)


# Set up each agent's objective function: x'Qx + u'Ru
Q = zeros(4, 4)   # All variables are unconstrained
R = I(2)


# Set up system properties: time horoizon and control bounds
N = 20
control_bounds = [-2.0, 2.0]
params = MPCParams(Q, R, system, control_bounds, N)


# Set up communication pattern: triangular sheaf
c = CellularSheaf([4, 4, 4], [2, 2, 2])
set_edge_maps!(c, 1, 2, 1, [4 0 0 0; 0 0 1 0], [2 0 0 0; 0 0 2 0])
# set_edge_maps!(c, 1, 3, 2, [2 0 0 0; 0 0 2 0], [1 0 0 0; 0 0 1 0])
set_edge_maps!(c, 2, 3, 3, [2 0 0 0; 0 0 2 0], [1 0 0 0; 0 0 4 0])


# Set up solver
x_init = BlockArray(5 * rand(12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params, params, params], c, x_init)
alg = ADMM(2.0, 10)  
num_iters = 100


# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)


# Plot results
PaperPlotting.paper_plot_save_results(trajectory, C, "Other", 1, "unconstrained, on the curve xy = 1")




# TEST CASE 2: Formation, unconstrained, on circular arcs

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)


# Set up each agent's objective function: x'Qx + u'Ru
Q = zeros(4, 4)   # All variables are unconstrained
R = I(2)


# Set up system properties: time horoizon and control bounds
N = 20
control_bounds = [-2.0, 2.0]
params = MPCParams(Q, R, system, control_bounds, N)


# Set up communication pattern: triangular sheaf
c = CellularSheaf([4, 4, 4], [2, 2, 2])
θ = π / 10
set_edge_maps!(c, 1, 2, 1, [cos(θ) 0 -sin(θ) 0; sin(θ) 0 cos(θ) 0], [1 0 0 0; 0 0 1 0])
set_edge_maps!(c, 1, 3, 2, [cos(θ / 2) 0 -sin(θ / 2) 0; sin(θ / 2) 0 cos(θ / 2) 0], [1 0 0 0; 0 0 1 0])
# set_edge_maps!(c, 2, 3, 3, [1 0 0 0; 0 0 1 0], [1 0 0 0; 0 0 1 0])


# Set up solver
x_init = BlockArray(5 * rand(12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params, params, params], c, x_init)
alg = ADMM(2.0, 10)  
num_iters = 100


# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)


# Plot results
PaperPlotting.paper_plot_save_results(trajectory, C, "Other", 2, "unconstrained, on circular arcs")