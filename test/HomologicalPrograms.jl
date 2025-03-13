using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots

# Discretization step size
dt = 0.1

# Set up each agent
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]

system = DiscreteLinearSystem(A, B, C)

#Q = I(4)
#Q[1, 1] = 0
Q = zeros(4, 4)

R = I(2)

N = 10

control_bounds = [-2.0, 2.0]

params = MPCParams(Q, R, system, control_bounds, N)

# Set up communication pattern
c = CellularSheaf([4, 4, 4], [2, 2, 2])

set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)

x_init = BlockArray(5 * rand(12), c.vertex_stalks)

prob = MultiAgentMPCProblem([params, params, params], c, x_init)
alg = ADMM(2.0, 10)

trajectory, controls = do_mpc!(prob, alg, 30)

agent_1_trajectory = mapreduce(permutedims, vcat, [C * x[Block(1)] for x in trajectory])
agent_2_trajectory = mapreduce(permutedims, vcat, [C * x[Block(2)] for x in trajectory])
agent_3_trajectory = mapreduce(permutedims, vcat, [C * x[Block(3)] for x in trajectory])


p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]])
plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]])
plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]])



#=x = BlockArray(rand(12), c.vertex_stalks)

y = nearest_section(c, x)

average_x = (x[Block(1)] + x[Block(2)] + x[Block(3)]) / 3

@test y[Block(1)] - average_x ≈ zeros(4) atol = 1e-5=#
