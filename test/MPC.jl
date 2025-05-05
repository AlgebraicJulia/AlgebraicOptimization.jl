using Test
using AlgebraicOptimization
using LinearAlgebra
using JuMP
using Plots

# Discretization step size
dt = 0.1

A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]
C = [1 0 0 0; 0 0 1 0]

system = DiscreteLinearSystem(A, B, C)

Q = I(4)
R = I(2)

N = 100

model = lq_tracking_model(Q, R, system, [10 * rand(), 5 * rand(), 10 * rand(), 5 * rand()], [5.0, 0.0, -2.0, 0.0], N, [-1, 1])

optimize!(model)

xs = value.(model[:x])

plot(xs[1, :], xs[3, :])

