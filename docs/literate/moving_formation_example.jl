# # Moving Formation Example
#
# This example combines a formation goal in positions with a 
# consensus goal in velocities. As such, the coordination sheaf 
# is the constant sheaf $\underline{\R}^4$ on the three vertex path graph. 
# This encodes a leader-follower topology with the middle agent 
# in the path acting as the leader. The leader’s objective is 
# to track a constant rightward velocity vector and minimize 
# its control actuation while the followers’ objectives are to 
# simply minimize control actuation. The edge potential functions 
# are of the form $U_e(y)=(1/2)\|y-b_e\|_2^2$ where the velocity 
# coordinates of each be are 0 encoding consensus in velocity while 
# the position coordinates are chosen to achieve a fixed displacement 
# between the leader and its followers. The results of this
# controller run for 160 iterations are shown below.
#
using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
include("../../../examples/paper-examples/PaperPlotting.jl")
using .PaperPlotting

# Set up each agent's dynamics: $x(t+1) = Ax(t) + Bu(t)$

dt = 0.1 # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)

# Initialize the weight matrices such that the objective only concerns velocities

Q_leader = [0 0 0 0; 0 50 0 0; 0 0 0 0; 0 0 0 50]
Q_follower = zeros(4, 4)
R = I(2)

# Define the parameters for the MPC

N = 10
control_bounds = [-2.0, 2.0]
params1 = MPCParams(Q_leader, R, system, control_bounds, N, [0.0, 3.0, 0.0, 0.0]) # track this velocity
params2 = params3 = MPCParams(Q_follower, R, system, control_bounds, N)

# Define the constant sheaf

c = CellularSheaf([4, 4, 4], [4, 4])
set_edge_maps!(c, 1, 2, 1, I(4), I(4))
set_edge_maps!(c, 1, 3, 2, I(4), I(4))


# Set up solver to perform MPC and solve the optimization problem with ADMM

x_init = BlockArray(5 * rand(-1.0:0.1:1.0, 12), c.vertex_stalks)
b = [5, 0, 5, 0, -5, 0, 5, 0] # Desired pairwise distance
prob = MultiAgentMPCProblem([params1, params2, params3], c, x_init, b)
alg = ADMM(2.0, 10)
num_iters = 160

# Run solver on MPC

trajectory, controls = do_mpc!(prob, alg, num_iters)

# Plot results with triangles

PaperPlotting.plot_trajectories(trajectory, C, false, true)