# Implementing block coordinate descent for homological programs.
# To start, we are assuming that each restriction map is full rank, so that
# we don't run into issues with the subproblems being ill-defined.

# The basic loop will be to iterate through the nodes in a cellular sheaf,
# solving the subproblem for each node in turn, and then passing the solution
# to the next node as the initial condition for the next subproblem.
# This will be done until convergence, or until a maximum number of iterations is reached.

using LinearAlgebra
using BlockArrays
using Optim
using Plots
using AlgebraicOptimization
using Test
using ForwardDiff

# Let's start by defining a simple example problem to test our implementation.

# Parameters for the example problem
subproblem_dim = 10 # Dimension of each subproblem
overlap_dim = 4 # Dimension of the overlap between subproblems
n_subproblems = 2 # Number of subproblems

# Make the restriction map as a projection from the subproblem to the overlap
R = zeros(overlap_dim, subproblem_dim)
for i in 1:overlap_dim
    R[i, i] = 1.0 # Identity in the overlap
end

#=s = @cellular_sheaf R begin
    x::Stalk{10}, y::Stalk{10} # Unrelated TODO: apparently this macro can't take variables as arguments, so we have to use a constant here

    R(x) == R(y)
end=#

s = CellularSheaf([subproblem_dim, subproblem_dim])
set_edge_maps!(s, 1, 2, R, R)

function newton_equality_constraint(f, A, b; x0=randn(size(A, 2)), ϵ=1e-6, max_iters=5, step_size=1)
    x = x0
    trace = [x]

    #println(x0)

    for iter in 1:max_iters
        g = ForwardDiff.gradient(f, x)
        H = ForwardDiff.hessian(f, x)

        KKT = [H A'; A zeros(size(A, 1), size(A, 1))]
        #rhs = [-g; zeros(size(A, 1))]
        rhs = [-g; -A * x + b]


        sol = KKT \ rhs
        Δx = sol[1:length(x)]
        λ = sol[length(x)+1:end]

        λ2 = Δx' * H * Δx
        if λ2 / 2 < ϵ
            println("Convergence in $iter iterations")
            return x
        end

        x += step_size * Δx
        push!(trace, x)
    end

    println("No convergence")
    return x
end



# Ok now let's define the coordinate descent algorithm
HP = HomologicalProgram{Function,CellularSheaf}

# This code has the right form, but we need to update the optimization step to account for the equality constraints
# coming from the neighboring nodes. Until we do that, the test at the end will fail.
function block_coordinate_descent(p::HP, max_iters::Int=10, tol::Float64=1e-6)
    # Initialize the solution
    x = BlockArray(ones(sum(p.sheaf.vertex_stalks)), p.sheaf.vertex_stalks)

    for iter in 1:max_iters
        r = 2^iter - 1
        x_prev = deepcopy(x) # Store the previous solution for convergence check

        # Iterate over each node in the sheaf
        for i in 1:length(p.objectives)
            # Solve the subproblem for node i
            objective = p.objectives[i]
            # This code has the right outline, but this step needs to be modified
            # to incorporate the equality constraint coming from the neighboring nodes.

            # Build the A matrix and b vector to pass to Newton's
            ns = neighbor_edges(p.sheaf, i)
            As = AbstractArray[]
            bs = Float64[]
            A, b = undef, undef
            for e in ns
                if e.v1 == i
                    A = e.rm12
                    B = e.rm21
                    y = x[Block(e.v2)]
                    b = B * y
                else
                    A = e.rm21
                    B = e.rm12
                    y = x[Block(e.v1)]
                    b = B * y
                end
                push!(As, A)
                bs = vcat(bs, b)
            end
            bigA = vcat(As...)

            #sol = newton_equality_constraint(objective, bigA, bs)
            f(x) = objective(x) + r * norm(bigA * x - bs)^2

            sol = optimize(f, x[Block(i)], LBFGS(); autodiff=:forward).minimizer
            x[Block(i)] = sol
        end

        # Check convergence
        if norm(x - x_prev) < tol
            println("Converged after $iter iterations")
            break
        end
    end

    return x
end



# Test on our example problem
objective1(x) = (x .- 1.0)' * (x .- 1.0) # Simple quadratic objective for the first subproblem
# ^minimized at x = 1
objective2(x) = (x .- 2)' * (x .- 2) # Simple quadratic objective for the second subproblem
# ^minimized at x = 2

p = HP([objective1, objective2], s)
solution = block_coordinate_descent(p)

# Test that we get a global section as the solution
#is_global_section(p.sheaf, solution)

