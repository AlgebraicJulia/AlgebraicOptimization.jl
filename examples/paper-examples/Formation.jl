using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables
using .PaperPlotting

# Number of agents (change as needed)
N_AGENTS = 5

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]
system = DiscreteLinearSystem(A, B, C)

# Set up each agent's objective function: x'Qx + u'Ru
Q = zeros(4, 4)
Q[1, 1] = 1    # x goes to 0
Q[3, 3] = 1    # y goes to 0
R = I(2)

N = 20
control_bounds = [-2.0, 2.0]

# TEST CASE 1: All agents go to 0
params = [MPCParams(Q, R, system, control_bounds, N) for _ in 1:N_AGENTS]

# Sheaf: star graph (agent 1 is the hub)
vertex_stalks = fill(4, N_AGENTS)
edge_stalks = fill(2, N_AGENTS - 1)
c = CellularSheaf(vertex_stalks, edge_stalks)
for i in 2:N_AGENTS
    set_edge_maps!(c, 1, i, i - 1, C, C)
end

# Set up solver
x_init = BlockArray(5 * rand(4 * N_AGENTS), vertex_stalks)
b = BlockArray(repeat([0.0, 0.0], N_AGENTS - 1), edge_stalks)
prob = MultiAgentMPCProblem(params, c, x_init, b)
alg = ADMM(2.0, 10)
num_iters = 100

# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)

# Plot results
PaperPlotting.paper_plot_save_results(
    trajectory,
    C,
    "Formation",
    1;
    additonal_str="all variables set to go to 0",
    n_agents=N_AGENTS
)
PaperPlotting.animate_trajectories_save_results(
    trajectory,
    C,
    "Formation",
    1;
    additonal_str="all variables set to go to 0",
    n_agents=N_AGENTS
)










# TEST CASE 2: "Follow the leader" (agent 1 goes to 0, others unconstrained)
params_leader = MPCParams(Q, R, system, control_bounds, N)
params_free = MPCParams(zeros(4, 4), R, system, control_bounds, N)
params2 = [params_leader; [params_free for _ in 2:N_AGENTS]...]

vertex_stalks2 = fill(4, N_AGENTS)
edge_stalks2 = fill(2, N_AGENTS - 1)
c2 = CellularSheaf(vertex_stalks2, edge_stalks2)
for i in 2:N_AGENTS
    set_edge_maps!(c2, 1, i, i - 1, C, C)
end

x_init2 = BlockArray(5 * rand(4 * N_AGENTS), vertex_stalks2)
b2 = BlockArray(repeat([0.0, 0.0], N_AGENTS - 1), edge_stalks2)
prob2 = MultiAgentMPCProblem(params2, c2, x_init2, b2)
alg2 = ADMM(2.0, 10)
num_iters2 = 200

trajectory2, controls2 = do_mpc!(prob2, alg2, num_iters2)

PaperPlotting.paper_plot_save_results(
    trajectory2,
    C,
    "Formation",
    2;
    additonal_str="one agent goes to (0, 0)",
    n_agents=N_AGENTS
)
PaperPlotting.animate_trajectories_save_results(
    trajectory2,
    C,
    "Formation",
    2;
    additonal_str="one agent goes to (0, 0)",
    n_agents=N_AGENTS
)














# TEST CASE 2: "Follow the leader" (agent 1 goes to 0, others unconstrained)
params_leader = MPCParams(Q, R, system, control_bounds, N)
params_follower = MPCParams(zeros(4, 4), R, system, control_bounds, N)
params2 = [params_leader; [params_follower for _ in 2:N_AGENTS]...]

# Sheaf: star graph (agent 1 is the hub)
vertex_stalks2 = fill(4, N_AGENTS)
edge_stalks2 = fill(2, N_AGENTS - 1)
c2 = CellularSheaf(vertex_stalks2, edge_stalks2)
for i in 2:N_AGENTS
    set_edge_maps!(c2, 1, i, i - 1, C, C)
end

x_init2 = BlockArray(5 * rand(4 * N_AGENTS), vertex_stalks2)
prob2 = MultiAgentMPCProblem(params2, c2, x_init2, vcat([5, 5, -5, 5] for _ in 1:N_AGENTS)...)
alg2 = ADMM(2.0, 10)
num_iters2 = 100

trajectory2, controls2 = do_mpc!(prob2, alg2, num_iters2)

PaperPlotting.paper_plot_save_results(
    trajectory2,
    C,
    "Formation",
    2;
    additonal_str="one agent goes to (0, 0)",
    follow_leader=true,
    n_agents=N_AGENTS
)
PaperPlotting.animate_trajectories_save_results(
    trajectory2,
    C,
    "Formation",
    2;
    additonal_str="one agent goes to (0, 0)",
    follow_leader=true,
    n_agents=N_AGENTS
)