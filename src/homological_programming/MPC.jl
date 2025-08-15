module MPC

export DiscreteLinearSystem, optimize_step, lqr_model, lq_tracking_model

using JuMP
using Ipopt
using LinearAlgebra

struct DiscreteLinearSystem
    A::AbstractMatrix
    B::AbstractMatrix
    C::AbstractMatrix     # Is this C actually used anywhere?
end

function (s::DiscreteLinearSystem)(x, u)
    return s.A * x + s.B * u
end


function lqr_model(Q::AbstractMatrix, R::AbstractMatrix, s::DiscreteLinearSystem, x0, x_target, horizon, control_bounds, ρ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppress solver output

    state_dim = size(s.A)[2]
    control_dim = size(s.B)[2]

    @assert size(Q) == (state_dim, state_dim)
    @assert size(R) == (control_dim, control_dim)

    @variable(model, x[1:state_dim, 1:horizon])
    @variable(model, control_bounds[1] <= u[1:control_dim, 1:horizon-1] <= control_bounds[2])

    @constraint(model, x[:, 1] .== x0)

    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== s.A * x[:, k] + s.B * u[:, k])
    end

    @objective(model, Min, sum((x[:, k]' * Q * x[:, k] + u[:, k]' * R * u[:, k]) for k in 1:horizon-1) + (ρ / 2) * (x[:, horizon] - x_target)' * (x[:, horizon] - x_target))

    return model
end

function lq_tracking_model(Q::AbstractMatrix, R::AbstractMatrix, s::DiscreteLinearSystem, x0, x_target, dual_target, horizon, control_bounds, ρ)
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppress solver output

    state_dim = size(s.A)[2]
    control_dim = size(s.B)[2]

    @assert size(Q) == (state_dim, state_dim)
    @assert size(R) == (control_dim, control_dim)

    @variable(model, x[1:state_dim, 1:horizon])
    @variable(model, control_bounds[1] <= u[1:control_dim, 1:horizon-1] <= control_bounds[2])

    @constraint(model, x[:, 1] .== x0)

    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== s.A * x[:, k] + s.B * u[:, k])
    end

    @objective(model, Min, sum(((x[:, k] - x_target)' * Q * (x[:, k] - x_target) + u[:, k]' * R * u[:, k]) for k in 1:horizon-1) + ρ * (x[:, horizon] - dual_target)' * (x[:, horizon] - dual_target))

    return model
end




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
function optimize_step(x_k, Q, R, s::DiscreteLinearSystem, x_target, ρ::Real)
    # Constants
    horizon = 10  # Prediction horizon

    # Define the optimization model using Ipopt solver
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppress solver output

    # Decision variables: state trajectory (x) and control inputs (u)
    @variable(model, x[1:2, 1:horizon])
    #@variable(model, u[1:2, 1:horizon])
    @variable(model, -20 <= u[1:2, 1:horizon] <= 20)  # Control limits

    # Initial state and control constraints
    @constraint(model, x[:, 1] .== x_k)
    #@constraint(model, u[:, 1] .== u_k)

    # System dynamics constraints: x[k+1] = Ax[k] + Bu[k]
    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== s.A * x[:, k] + s.B * u[:, k])
    end

    # Define the cost function (sum of squared states and inputs over the horizon)
    #@objective(model, Min, sum((x[:, k]' * Q * x[:, k]) + (u[:, k]' * R * u[:, k]) for k = 1:horizon))# +  5 * ((x[:, horizon] - x_target)' * Q * (x[:, horizon] - x_target)))
    @objective(model, Min, sum((x[:, k]' * Q * x[:, k]) + (u[:, k]' * R * u[:, k]) for k = 1:horizon)) + ρ / 2 * ((x[:, horizon] - x_target)' * Q * (x[:, horizon] - x_target))
    #@objective(model, Min, sum((x[:, k]' * Q * x[:, k]) + (u[:, k]' * R * u[:, k]) for k = 1:horizon))
    # Solve the optimization problem
    optimize!(model)

    # Return the optimized control input for the next time step
    return value.(x[:, horizon]), value.(u[:, 1])
end


end # module