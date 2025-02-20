using JuMP
using Ipopt

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
    @objective(model, Min, sum(x[:, k]'*x[:, k] for k = 1:horizon))

    # Solve
    optimize!(model)

    return value.(u[:, 2])
end


function do_mpc(x_0, u_0)
    x = x_0
    u = u_0

    for i in 1:20
        u = simple_mpc(x, u)
        x += u
        println("i = ", i)
        println("x = ", x)
    end
end
    

    


x = [1.0; 2.0]
u = [3.0, 4.0]
simple_mpc(x, u)

do_mpc(x, u)