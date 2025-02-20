using JuMP
using Ipopt
using Plots
using PlotThemes

# Simple MPC where x_k is a matrix of the current state and u_k is a matrix of the current input
function simple_mpc(x_k, u_k)
    # Constants
    horizon = 10

    # Model
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Variables
    @variable(model, x[1:2, 1:horizon])
    @variable(model, -1 <= u[1:2, 1:horizon] <= 1)
    
    # Constraints
    @constraint(model, x[:, 1] .== x_k)
    @constraint(model, u[:, 1] .== u_k)
    for k = 1:horizon-1
        @constraint(model, x[:, k+1] .== x[:, k] + u[:, k])
    end

    # Objective
    @objective(model, Min, sum(x[:, k]'*x[:, k] + u[:, k]'*u[:, k] for k = 1:horizon))

    # Solve
    optimize!(model)

    return value.(u[:, 2])
end


function do_mpc(x_0, u_0)
    x = [x_0]  # Store x as a list of vectors
    u = u_0

    for i in 1:99
        u = simple_mpc(x[end], u)  # Use the last state
        new_x = x[end] + u  # Compute next state
        push!(x, new_x)  # Store it
        println("i = ", i)
        println("u = ", u)
        println("x = ", new_x)
    end

    # Convert x to a matrix for plotting
    x_matrix = hcat(x...)  # Stack states as columns

    # Plot each state variable

    theme(:juno)

    p = plot(1:100, x_matrix', label=["x1" "x2"])

    savefig(p, "./examples/single_agent_mpc.png")

end
    

    


x = [1.0; 2.0]
u = [3.0, 4.0]
simple_mpc(x, u)

do_mpc(x, u)