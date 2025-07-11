using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables
include("PaperPlotting.jl")
using .PaperPlotting

# Number of agents
N_AGENTS = 12

# Set up each agent's dynamics: x' = Ax + Bu
dt = 0.1  # Discretization step size
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0]  # Only x in consensus
system = DiscreteLinearSystem(A, B, C)

# Set up each agent's objective function: x'Qx + u'Ru
Q = I(4)
Q[1, 1] = 0    # First variable is unconstrained
R = I(2)

# Set up system properties: time horizon and control bounds
N = 10
control_bounds = [-2.0, 2.0]
targets = [[0, 0, 2.0 * i, 0] for i in 1:N_AGENTS]
params = [MPCParams(Q, R, system, control_bounds, N, targets[i]) for i in 1:N_AGENTS]

# Set up communication pattern: fully connected sheaf
vertex_stalks = fill(4, N_AGENTS)
edge_stalks = fill(1, N_AGENTS * (N_AGENTS - 1) ÷ 2)
c = CellularSheaf(vertex_stalks, edge_stalks)

function edge_indexing()
    edge_idx = 1
    for i in 1:N_AGENTS-1
        for j in i+1:N_AGENTS
            set_edge_maps!(c, i, j, edge_idx, C, C)
            edge_idx += 1
        end
    end
end
edge_indexing()

# Set up solver
x_init = BlockArray(5 * rand(-1:0.1:1, 4 * N_AGENTS), c.vertex_stalks)
prob = MultiAgentMPCProblem(params, c, x_init)
alg = ADMM(2.0, 5)
num_iters = 100

# Run solver
trajectory, controls = do_mpc!(prob, alg, num_iters)

# Plot results (show only first 10 agents for clarity)
PaperPlotting.paper_plot_save_results(
    trajectory,
    [1 0 0 0; 0 0 1 0],
    "Consensus",
    2;
    additonal_str="x unconstrained and y constrained",
    n_agents=N_AGENTS
)



animate_trajectories(trajectory, [1 0 0 0; 0 0 1 0]; n_agents=N_AGENTS)


PaperPlotting.animate_trajectories_save_results(
    trajectory,
    [1 0 0 0; 0 0 1 0],
    "Consensus",
    2;
    additonal_str="x unconstrained and y constrained",
    n_agents=N_AGENTS
)