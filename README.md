# AlgebraicOptimization.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://AlgebraicJulia.github.io/AlgebraicOptimization.jl/stable)
[![Development Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://AlgebraicJulia.github.io/AlgebraicOptimization.jl/dev)
[![Code Coverage](https://codecov.io/gh/AlgebraicJulia/AlgebraicOptimization.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AlgebraicJulia/AlgebraicOptimizatione.jl)
[![CI/CD](https://github.com/AlgebraicJulia/AlgebraicOptimization.jl/actions/workflows/julia_ci.yml/badge.svg)](https://github.com/AlgebraicJulia/AlgebraicOptimization.jl/actions/workflows/julia_ci.yml)

This package is designed for building large optimization problems out of simpler subproblems and automatically compiling them to a distributed solver.

## Basic Usage

The most simple use of this package is making and solving a composite optimization problem. For example, say we have three subproblems whose objectives are given by some random quadratic cost functions:
```julia
using AlgebraicOptimization
using Catlab

# Problem parameters.
P = rand(-1:0.01:1,5,5)
P = P'*P
Q = rand(-1:0.01:1,3,3)
Q = Q'*Q
R = rand(-1:0.01:1,4,4)
R = R'*R

a = rand(-1:0.01:1,5)
b = rand(-1:0.01:1,3)
c = rand(-1:0.01:1,4)

# Subproblem objectives.
# A PrimalObjective wraps an objective function with its input dimension.
f = PrimalObjective(FinSet(5),x->x'*P*x + a'*x)
g = PrimalObjective(FinSet(3),x->x'*Q*x + b'*x)
h = PrimalObjective(FinSet(4),x->x'*R*x + c'*x)
```

Now, to compose these subproblems, we need to make them into *open* problems. An open problem specifies which components of a problem's domain are open to composition with other problems. We do this as follows:
```julia
# Make open problems.
# The first argument is the primal objective we are wrapping, the second argument is a function
# specifying which components of the objective are exposed. 
p1 = Open{PrimalObjective}(FinSet(5), PrimalObjective(FinSet(5),f), FinFunction([2,4], 5))
p2 = Open{PrimalObjective}(FinSet(3), PrimalObjective(FinSet(3),g), id(FinSet(3)))
p3 = Open{PrimalObjective}(FinSet(4), PrimalObjective(FinSet(4),h), FinFunction([1,3,4]))
```

To specify the composition pattern of our subproblems, we use Catlab's relation macro to make an undirected wiring diagram and `oapply` to compose our subproblems.
```julia
d = @relation_diagram (x,y,z) begin
    f(u,x)
    g(u,w,y)
    h(u,w,z)
end

composite_prob = oapply(d, [p1,p2,p3])
```

Now, we can solve the composite problem with distributed gradient descent:
```julia
# Arguments are problem, initial guess, step size, and number of iterations
sol = solve(composite, repeat([100.0], dom(composite_prob).n), 0.1, 100)
```
Currently, we just support unconstrained and equality constrained problems with plans to add support for inequality constrained and disciplined convex programs.

More complete documentation and quality of life improvements are also on the way.

This package is an implementation of [A Compositional Framework for First-Order Optimization](https://arxiv.org/abs/2403.05711).


