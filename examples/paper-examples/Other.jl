using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables



# TEST CASE 4: Formation, unconstrained, on the curve xy = 1

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


# Plot results    TODO: Modularize this code
agent_1_trajectory = mapreduce(permutedims, vcat, [C * x[Block(1)] for x in trajectory])
agent_2_trajectory = mapreduce(permutedims, vcat, [C * x[Block(2)] for x in trajectory])
agent_3_trajectory = mapreduce(permutedims, vcat, [C * x[Block(3)] for x in trajectory])

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2], labels="", color=:red)
scatter!(agent_1_trajectory[2:end, 1], agent_1_trajectory[2:end, 2], label="Agent 1", color=:red)
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]], label="", color=:cyan)

plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2], labels="", color=:blue)
scatter!(agent_2_trajectory[2:end, 1], agent_2_trajectory[2:end, 2], label="Agent 2", color=:blue)
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]], label="", color=:cyan)

plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2], labels="", color=:green)
scatter!(agent_3_trajectory[2:end, 1], agent_3_trajectory[2:end, 2], label="Agent 3", color=:green)
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]], label="Initial Positions", color=:cyan)

title!("Formation (unconstrained, on the curve xy = 1)")
xlabel!("x-position")
ylabel!("y-position")









# TEST CASE 5: Formation, unconstrained, on circular arcs

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


# Plot results    TODO: Modularize this code
agent_1_trajectory = mapreduce(permutedims, vcat, [C * x[Block(1)] for x in trajectory])
agent_2_trajectory = mapreduce(permutedims, vcat, [C * x[Block(2)] for x in trajectory])
agent_3_trajectory = mapreduce(permutedims, vcat, [C * x[Block(3)] for x in trajectory])

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2], labels="", color=:red)
scatter!(agent_1_trajectory[2:end, 1], agent_1_trajectory[2:end, 2], label="Agent 1", color=:red)
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]], label="", color=:cyan)

plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2], labels="", color=:blue)
scatter!(agent_2_trajectory[2:end, 1], agent_2_trajectory[2:end, 2], label="Agent 2", color=:blue)
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]], label="", color=:cyan)

plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2], labels="", color=:green)
scatter!(agent_3_trajectory[2:end, 1], agent_3_trajectory[2:end, 2], label="Agent 3", color=:green)
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]], label="Initial Positions", color=:cyan)

title!("(Formation) unconstrained, on circular arcs")
xlabel!("x-position")
ylabel!("y-position")