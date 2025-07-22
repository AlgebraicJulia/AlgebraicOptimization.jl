using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables
include("PaperPlotting.jl")
using .PaperPlotting

# Number of agents (change as needed)
N_AGENTS = 6

# Set up each agent's dynamics: x(t+1) = Ax(t) + Bu(t)
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)

# Leader and follower cost matrices
Q_leader = [0 0 0 0; 0 50 0 0; 0 0 0 0; 0 0 0 50] # Objective only concerns velocities
Q_follower = zeros(4, 4)
R = I(2)
# Something new
R_follower = zeros(2, 2)

N = 10
control_bounds = [-2.0, 2.0]

# Leader tracks a velocity, followers have zero cost (formation enforced by sheaf)
params = [MPCParams(Q_leader, R, system, control_bounds, N, [0.0, 1.0, 0.0, 0.0])]
for i in 2:N_AGENTS
    push!(params, MPCParams(Q_follower, R_follower, system, control_bounds, N))
end

# Potential function for formation (same for all agents)
q(x) = (x' * x - 5.0)^2

# Constant sheaf for formation (generalized for N_AGENTS)
vertex_stalks = fill(4, N_AGENTS)
edge_stalks = fill(2, N_AGENTS * (N_AGENTS - 1) ÷ 2)
potentials = [q for _ in 1:N_AGENTS]
c = PotentialSheaf(vertex_stalks, edge_stalks, potentials)

# # Set edge maps (fully connected for formation)
# edge_idx = 1
# for i in 1:N_AGENTS-1
#     for j in i+1:N_AGENTS
#         set_edge_maps!(c, i, j, edge_idx, C, C)
#         edge_idx += 1
#     end
# end

# Set edge maps (star graph: leader connects to every other agent)
for i in 2:N_AGENTS
    set_edge_maps!(c, 1, i, i - 1, C, C)
end

# Set up solver
x_init = BlockArray(rand(4 * N_AGENTS), c.vertex_stalks)
prob = MultiAgentMPCProblem(params, c, x_init)
alg = NonConvexADMM(1000.0, 10, 0.0001, 5000)
num_iters = 200

# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)

# Plot results
PaperPlotting.plot_trajectories(trajectory, C; n_agents=N_AGENTS)
#PaperPlotting.paper_plot_save_results(trajectory, C, "Flocking", 6, "Fixed Distances", n_agents=N_AGENTS)

PaperPlotting.animate_trajectories_save_results(
    trajectory, C, "Flocking", 6;
    additonal_str="Fixed Distances",
    n_agents=N_AGENTS
)



#PaperPlotting.paper_plot_save_results(trajectory, C, "Flocking", 6, "Fixed Distances")

#=
# Set up each agent's dynamics: x(t+1) = Ax(t) + Bu(t)
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)

# TEST CASE 1: Flocking. Use the constant sheaf on the communication topology.
# Consensus on velocities and formation on positions.

Q_leader = [0 0 0 0; 0 50 0 0; 0 0 0 0; 0 0 0 50] # Objective only concerns velocities
Q_follower = zeros(4, 4)
R = I(2)

T = 10

control_bounds = [-2.0, 2.0]
params1 = MPCParams(Q_leader, R, system, control_bounds, T, [0.0, 3.0, 0.0, 0.0]) # track this velocity
params2 = params3 = MPCParams(Q_follower, R, system, control_bounds, T)

# Constant sheaf
c = CellularSheaf([4, 4, 4], [4, 4])
set_edge_maps!(c, 1, 2, 1, I(4), I(4))
set_edge_maps!(c, 1, 3, 2, I(4), I(4))
#set_edge_maps!(c, 2, 3, 3, I(4), I(4))


# Set up solver
x_init = BlockArray(5 * rand(-1.0:0.1:1.0, 12), c.vertex_stalks)
b = [5, 0, 5, 0, -5, 0, 5, 0]
prob = MultiAgentMPCProblem([params1, params2, params3], c, x_init, b)
alg = ADMM(2.0, 10)
num_iters = 160


# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)

# Save results to CSVs
path = "./examples/paper-examples/flocking/3/"
experiment = "flocking"

agent_trajectories = PaperPlotting.postprocess_trajectory(trajectory, [C, C, C])

PaperPlotting.save_trajectories(path, experiment, agent_trajectories)


# Plot results
#PaperPlotting.paper_plot_save_results(trajectory, C, "Flocking", 2, "Consensus in velocity, Formation in position")

# TEST CASE 2: "Accidental flocking". Consensus, with all variables unconstrained. Looks similar to flocking.

#=
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


# Plot results
PaperPlotting.paper_plot_save_results(trajectory, C, "Flocking", 1, "accidental, all variables unconstrained")
=#
=#