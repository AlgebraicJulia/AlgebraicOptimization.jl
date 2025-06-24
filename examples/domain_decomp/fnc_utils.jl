# Utility functions taken from the Fundamentals of Numerical Computation Textbook https://fncbook.com/

"""
    diffmat2(n,xspan)

Compute 2nd-order-accurate differentiation matrices on `n`+1 points
in the interval `xspan`. Returns a vector of nodes and the matrices
for the first and second derivatives.
"""
function diffmat2(n, xspan)
    a, b = xspan
    h = (b - a) / n
    x = [a + i * h for i in 0:n]   # nodes

    # Define most of Dₓ by its diagonals.
    dp = fill(0.5 / h, n)        # superdiagonal
    dm = fill(-0.5 / h, n)       # subdiagonal
    Dₓ = diagm(-1 => dm, 1 => dp)

    # Fix first and last rows.
    Dₓ[1, 1:3] = [-1.5, 2, -0.5] / h
    Dₓ[n+1, n-1:n+1] = [0.5, -2, 1.5] / h

    # Define most of Dₓₓ by its diagonals.
    d0 = fill(-2 / h^2, n + 1)    # main diagonal
    dp = ones(n) / h^2         # super- and subdiagonal
    Dₓₓ = diagm(-1 => dp, 0 => d0, 1 => dp)

    # Fix first and last rows.
    Dₓₓ[1, 1:4] = [2, -5, 4, -1] / h^2
    Dₓₓ[n+1, n-2:n+1] = [-1, 4, -5, 2] / h^2

    return x, Dₓ, Dₓₓ
end

"""
    bvplin(p, q, r, xspan, lval, rval, n)

Use finite differences to solve a linear bopundary value problem.
The ODE is u''+`p`(x)u'+`q`(x)u = `r`(x) on the interval `xspan`,
with endpoint function values given as `lval` and `rval`. There will
be `n`+1 equally spaced nodes, including the endpoints.

Returns vectors of the nodes and the solution values.
"""
function bvplin(γ, p, q, r, xspan, lval, rval, n)
    x, Dₓ, Dₓₓ = diffmat2(n, xspan)

    P = diagm(p.(x))
    Q = diagm(q.(x))
    L = γ * Dₓₓ + P * Dₓ + Q     # ODE expressed at the nodes

    # Replace first and last rows using boundary conditions.
    z = zeros(1, n)
    A = [[1 z]; L[2:n, :]; [z 1]]
    b = [lval; r.(x[2:n]); rval]

    # Solve the system.
    u = A \ b
    return x, u
end

function bvplin_solver(γ, p, q, r, xspan, n)
    x, Dₓ, Dₓₓ = diffmat2(n - 1, xspan)

    P = diagm(p.(x))
    Q = diagm(q.(x))
    L = γ * Dₓₓ + P * Dₓ + Q     # ODE expressed at the nodes

    # Replace first and last rows using boundary conditions.
    z = zeros(1, n - 1)
    A = [[1 z]; L[2:n-1, :]; [z 1]]
    println(size(A))
    rvec = r.(x[2:n-1])
    function solve(lval, rval)
        b = [lval; rvec; rval]
        # Solve the system.
        u = A \ b
        return x, u
    end
    return solve
end

