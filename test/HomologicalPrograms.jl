# OUTDATED! LOOK AT examples/paper-examples FOR ALL OF THIS CODE BUT UPDATED.


using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables

# TEST CASE 1: Consensus, with x unconstarined and y constarined.

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)


# Set up each agent's objective function: x'Qx + u'Ru
Q = I(4)
Q[1, 1] = 0    # First variable is unconstrained
R = I(2)


# Set up system properties: time horoizon and control bounds
N = 20
control_bounds = [-2.0, 2.0]
params = MPCParams(Q, R, system, control_bounds, N)


# Set up communication pattern: triangular sheaf
c = CellularSheaf([4, 4, 4], [2, 2, 2])
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)


# Set up solver
x_init = BlockArray(5 * rand(12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params, params, params], c, x_init)
alg = ADMM(2.0, 10)  
num_iters = 100


# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)


# Plot results
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

# savefig(p, "/examples/loop")

# CSV.write("./examples/ex1/traj1.csv", Tables.table(agent_1_trajectory))
# CSV.write("./examples/ex1/traj2.csv", Tables.table(agent_2_trajectory))
# CSV.write("./examples/ex1/traj3.csv", Tables.table(agent_3_trajectory))


# TEST CASE 2: Consensus, with all variables unconstrained. Looks similar to flocking.

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
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)


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

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]])
plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]])
plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]])








# TEST CASE 3: Consensus, all variables unconstrained, and non-trivial restriction maps.

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
set_edge_maps!(c, 1, 2, 1, [2 0 0 0; 0 0 2 0], [1 0 0 0; 0 0 1 0])
set_edge_maps!(c, 1, 3, 2, [2 0 0 0; 0 0 2 0], [1 0 0 0; 0 0 1 0])
set_edge_maps!(c, 2, 3, 3, [2 0 0 0; 0 0 2 0], [1 0 0 0; 0 0 1 0])


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

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]])
plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]])
plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]])







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

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]])
plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]])
plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]])









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

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]])
plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]])
plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]])






# TEST CASE 6: Consensus, all variables set to go to 0

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)


# Set up each agent's objective function: x'Qx + u'Ru
Q = zeros(4, 4)   # All variables are unconstrained
Q[1, 1] = 1    # x goes to 0
Q[3, 3] = 1    # y goes to 0
R = I(2)


# Set up system properties: time horoizon and control bounds
N = 20
control_bounds = [-2.0, 2.0]
params = MPCParams(Q, R, system, control_bounds, N)


# Set up communication pattern: triangular sheaf
c = CellularSheaf([4, 4, 4], [2, 2, 2])
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
set_edge_maps!(c, 2, 3, 3, C, C)


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

p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2])
scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]])
plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2])
scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]])
plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2])
scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]])







# TEST CASE 7: Consensus, one agent goes to (0, 0). "follow the leader"

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)


# Set up each agent's objective function: x'Qx + u'Ru
Q = zeros(4, 4)   # All variables are unconstrained
Q[1, 1] = 1    # x goes to 0
Q[3, 3] = 1    # y goes to 0
R = I(2)


# Set up system properties: time horoizon and control bounds
N = 20
control_bounds = [-2.0, 2.0]
params = MPCParams(Q, R, system, control_bounds, N)
params_free = MPCParams(zeros(4, 4), R, system, control_bounds, N)


# Set up communication pattern: triangular sheaf
c = CellularSheaf([4, 4, 4], [2, 2])
set_edge_maps!(c, 1, 2, 1, C, C)
set_edge_maps!(c, 1, 3, 2, C, C)
# set_edge_maps!(c, 2, 3, 3, C, C)   # Could comment out this line to make the 3rd agent not communicate with the 2nd


# Set up solver
x_init = BlockArray(5 * rand(12), c.vertex_stalks)
prob = MultiAgentMPCProblem([params, params_free, params_free], c, x_init)
alg = ADMM(2.0, 10)  
num_iters = 100


# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)


# Plot results    TODO: Modularize this code
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




# Sam's notes:

# Big question: why do these trajectories have 2 variables and not 4?
# Answer: we're only plotting variables 1 and 2.
# And I think it has to do with the mapreduce function



# Potential variants:

# Can we get the y's to not always go to 0?
# What are we actually looking at? Can we slap some axis labels on this?
# Could we use a larger step size or something? Becaues 100 iterations is a lot and it takes a while to run.
# Why are the x's going to a fixed something but the y's are going to 0?
# Rotation formation. I.e. We form like an arc w/ 3 rays
# I'm starting to feel like more agents would be helpful for the visual


# Defining A, B, and C:
# So the 1st and 3rd variables are shared with your neighbor?

# Isn't this C pretty different from the A and B? What's up with that? Since we're using C for the communication pattern,
                                    # and A and B for the individual agent dynamics.


                                    # How did we decide there would be 2 control variables?


# alg = ADMM(2.0, 10)   # Should we be using dt here?
