using Plots
using LinearAlgebra
using AlgebraicOptimization
using CSV, Tables
include("PaperPlotting.jl")
using .PaperPlotting

dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [0 1 0 0; 0 0 0 1]
system = DiscreteLinearSystem(A, B, C)

Q = zeros(4, 4)
R = I(2)

T = 10

control_bounds = [-2.0, 2.0]
params = MPCParams(Q, R, system, control_bounds, T)



c = CellularSheaf([4, 4, 4], [2, 2, 2])
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)

x_init = BlockArray(5 * rand(-1.0:0.1:1.0, 12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params, params, params], c, x_init)
alg = ADMM(2.0, 10)
num_iters = 160

trajectory, controls = do_mpc!(prob, alg, num_iters)

path = "./examples/paper-examples/flocking/4/"
experiment = "velocity-consensus"

C2 = [1 0 0 0; 0 0 1 0]

agent_trajectories = PaperPlotting.postprocess_trajectory(trajectory, [C2, C2, C2])

PaperPlotting.save_trajectories(path, experiment, agent_trajectories)