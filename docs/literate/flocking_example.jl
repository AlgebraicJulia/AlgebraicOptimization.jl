# # Flocking Example
#
# For this example, agents implement the
# standard flocking goal of reaching consensus in velocities
# while staying a fixed distance away from all other agents.
# The constant sheaf $\underline{\R}^4$ on a fully connected communication
# topology along with potential functions summing the stan-
# dard consensus potential function on the velocity components
# and the fixed distance potential function with $r^2 = 5$ on the
# position components. Each agents’ objective is to minimize
# total control activation. Additionally, a designated leader
# agent tracks a constant rightward velocity vector. The results
# of this controller run for 65 iterations are shown below.
# Computing the distance between each agent confirms that
# they reached the desired pairwise distance of $\sqrt{5}$.
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
params1 = MPCParams(Q_leader, R, system, control_bounds, N, [0.0, 1.0, 0.0, 0.0])
params2 = params3 = MPCParams(Q_follower, R, system, control_bounds, N)

# Define the potential functions
#
# $q(x) = (x' * x - 5.0)^2$
#
# $p(x) = [x[2], x[4]]' * [x[2], x[4]] + q([x[1], x[3]])$

q(x) = (x' * x - 5.0)^2
p(x) = [x[2], x[4]]' * [x[2], x[4]] + q([x[1], x[3]])

# Define the constant sheaf

c = PotentialSheaf([4, 4, 4], [2, 2, 2], [q, q, q])
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)

# Set up solver to perform MPC and solve the optimization problem with ADMM

x_init = BlockArray(rand(12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params1, params1, params1], c, x_init)
alg = NonConvexADMM(1000.0, 10, 0.0001, 5000)
num_iters = 200

# Run solver on MPC

trajectory, controls = do_mpc!(prob, alg, num_iters)


# Plot results

PaperPlotting.plot_trajectories(trajectory, C)