using JuMP
using Ipopt
using Plots
using PlotThemes
using LinearAlgebra

"""     optimize_step(x_k, u_k)

Performs a single Model Predictive Control (MPC) optimization step.

# Arguments
- `x_k::Matrix{Float64}`: The current state matrix (2x1 vector).
- `u_k::Matrix{Float64}`: The current input matrix (2x1 vector).
- `Q::Matrix{Float64}`: The state cost matrix (2x2).
- `R::Matrix{Float64}`: The input cost matrix (2x2).
- `x_target::Matrix{Float64}`: The target state matrix (2x1 vector).

# Returns
- `Vector{Float64}`: The optimized control input for the next step.
"""
function optimize_step(x_k, u_k, Q, R, x_target)
    # Constants
    horizon = 10  # Prediction horizon

    # Define the optimization model using Ipopt solver
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppress solver output

    # Decision variables: state trajectory (x) and control inputs (u)
    @variable(model, x[1:2, 1:horizon])
    #@variable(model, u[1:2, 1:horizon])
    @variable(model, -1 <= u[1:2, 1:horizon] <= 1)  # Control limits
    
    # Initial state and control constraints
    @constraint(model, x[:, 1] .== x_k)
    @constraint(model, u[:, 1] .== u_k)
    
    # System dynamics constraints: x[k+1] = Ax[k] + Bu[k]
    A = [1.0 0.05; 0.0 1.0]  # State matrix
    B = [0.05 0.0; 0.0 0.05]  # Input matrix
    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== A*x[:, k] + B*u[:, k])
    end

    # Define the cost function (sum of squared states and inputs over the horizon)
    #@objective(model, Min, sum((x[:, k]'*Q*x[:, k]) + (u[:, k]'*R*u[:, k]) for k = 1:horizon))# +  5 * ((x[:, horizon] - x_target)' * Q * (x[:, horizon] - x_target)))
    @objective(model, Min, sum(((x[:, k] - x_target)'*Q*(x[:, k] - x_target)) + (u[:, k]'*R*u[:, k]) for k = 1:horizon))

    # Solve the optimization problem
    optimize!(model)

    # Return the optimized control input for the next time step
    return value.(u[:, 2])
end

"""     do_mpc(x_0, u_0)

Runs Model Predictive Control (MPC) over multiple time steps and visualizes the state trajectory.

# Arguments
- `x_0::Vector{Float64}`: Initial state vector (2D system).
- `u_0::Vector{Float64}`: Initial control input vector (2D system).

"""
function do_mpc(x_0, u_0)
    x = [x_0]  # Store state trajectory as a list of vectors
    u_to_plot = [u_0]
    u = u_0  # Initial control input
    Q = 2*Matrix(I, 2, 2)  # State weights
    R = 2*Matrix(I, 2, 2)   # Control input weights
    # Q = randn(2,2)  # State weights
    # Q = Q * Q'  # Ensure Q is positive semi-definite
    # R = randn(2,2)  # Control input weights
    # R = R * R'  # Ensure R is positive semi-definite
    x_target = [7.0; 13.0]  # Desired terminal state

    # MPC loop for 99 iterations
    for i in 1:99
        u = optimize_step(x[end], u, Q, R, x_target)  # Compute optimal control input
        push!(u_to_plot, u)  # Store control input for plotting
        new_x = x[end] + u  # Update state using system dynamics
        push!(x, new_x)  # Store new state
    end

    # Plot results of state vs. time
    x_matrix = hcat(x...)
    theme(:juno)
    p = plot(1:100, x_matrix', label=["x1" "x2"], title="MPC State Evolution", 
    xlabel="Iteration", ylabel="State value")
    savefig(p, "./examples/single_agent_mpc.png")

    # Plot results of control input vs. time
    u_matrix = hcat(u_to_plot...)   
    plot!(1:100, u_matrix', label=["u1" "u2"], title="MPC Control Input Evolution",
    xlabel="Iteration", ylabel="Control input value")
    savefig(p, "./examples/single_agent_mpc_control.png")

    p = plot(x_matrix[1, :], x_matrix[2, :], label="State trajectory", title="MPC State Trajectory",
    xlabel="x1", ylabel="x2", legend=:bottomright)
    scatter!(x_matrix[1, :], x_matrix[2, :], label="States", marker=:circle, markersize=4)  # Overlay points

    savefig(p, "./examples/single_agent_mpc_trajectory.png")
end

# Example initial conditions for state and control input
x = vec(rand(1,2) * 10.0)
#u = vec(rand(1,2) * 10.0)
u = [0.0; 0.0]

# Run the MPC simulation and plot results
do_mpc(x, u)
