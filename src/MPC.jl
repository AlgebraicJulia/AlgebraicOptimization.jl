using JuMP
using Ipopt

# Simple MPC where x_k is a matrix of the current state and u_k is a matrix of the current input
function simple_mpc(x_k, u_k)
    # Constants
    horizon = 10

    # Model
    model = Model(solver=IpoptSolver(print_level=0))

    # Variables
    @variable(model, zeros((2, horizon)) <= x[1:2, 1:horizon])
    @variable(model, zeros((1, horizon)) <= u[1:1, 1:horizon])
    
    # Constraints
    @constraint(model, x[:, 1] .== x_k)
    @constraint(model, u[:, 1] .== u_k)
    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== x[:, k] + u[:, k])
    end

    # Objective
    @objective(model, Min, sum(x[:, k]'*x[:, k] + u[:, k]'*u[:, k] for k = 1:horizon))

    # Solve
    solve(model)

    return getvalue(u[:, 1])
end
    