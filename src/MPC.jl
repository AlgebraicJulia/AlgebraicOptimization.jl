using JuMP
using Ipopt
using Plots
using PlotThemes

"""     optimize_step(x_k, u_k)

Performs a single Model Predictive Control (MPC) optimization step.

# Arguments
- `x_k::Matrix{Float64}`: The current state matrix (2x1 vector).
- `u_k::Matrix{Float64}`: The current input matrix (2x1 vector).

# Returns
- `Vector{Float64}`: The optimized control input for the next step.
"""
function optimize_step(x_k, u_k)
    # Constants
    horizon = 10  # Prediction horizon

    # Define the optimization model using Ipopt solver
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppress solver output

    # Decision variables: state trajectory (x) and control inputs (u)
    @variable(model, x[1:2, 1:horizon])
    @variable(model, -1 <= u[1:2, 1:horizon] <= 1)  # Control limits
    
    # Initial state and control constraints
    @constraint(model, x[:, 1] .== x_k)
    @constraint(model, u[:, 1] .== u_k)
    
    # System dynamics constraints: x[k+1] = x[k] + u[k]
    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== x[:, k] + u[:, k])
    end
    
    # Define the cost function (sum of squared states and inputs over the horizon)
    @objective(model, Min, sum(x[:, k]'x[:, k] + u[:, k]'*u[:, k] for k = 1:horizon))

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
    u = u_0  # Initial control input

    # MPC loop for 99 iterations
    for i in 1:99
        u = optimize_step(x[end], u)  # Compute optimal control input
        new_x = x[end] + u  # Update state using system dynamics
        push!(x, new_x)  # Store new state
    end

    # Plot results of state vs. time
    x_matrix = hcat(x...)
    theme(:juno)
    p = plot(1:100, x_matrix', label=["x1" "x2"], title="MPC State Evolution")
    savefig(p, "./examples/single_agent_mpc.png")
end

# Example initial conditions for state and control input
x = [1.0; 2.0]
u = [3.0, 4.0]

# Run the MPC simulation and plot results
do_mpc(x, u)
