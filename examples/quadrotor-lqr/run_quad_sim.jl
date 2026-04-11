using AlgebraicOptimization
using Plots
using LinearAlgebra

edges   = [(1,2), (2,3)]
offsets = Dict((1,2) => [2.0, 0.0, 0.0],
               (2,3) => [2.0, 0.0, 0.0])

x0s       = [zeros(12) for _ in 1:3]
waypoints = [vcat([0.0, 0.0, 2.0], zeros(9)),
             vcat([2.0, 0.0, 2.0], zeros(9)),
             vcat([4.0, 0.0, 2.0], zeros(9))]

D, b = formation_coboundary(3, edges, offsets)

println("Running baseline simulation...")
base = run_baseline_sim(waypoints; x0s=x0s, t_end=10.0)

println("Running coordinated simulation...")
coord = run_coordinated_sim(3, edges, offsets; x0s=x0s, t_end=10.0)

err(recs) = norm(D * vcat([recs[i].x[1:3, end] for i in 1:3]...) - b)
println("\nFormation error at t = 10 s:")
println("  Baseline:    ", round(err(base),  sigdigits=4), " m")
println("  Coordinated: ", round(err(coord), sigdigits=4), " m")

println("\nSaving plots...")
savefig(compare_runs(base, coord, D, b), "examples/quadrotor-lqr/figures/comparison.png")
savefig(plot_trajectories(coord),        "examples/quadrotor-lqr/figures/trajectories.png")
savefig(plot_formation_error(coord, D, b), "examples/quadrotor-lqr/figures/formation_error.png")
savefig(plot_motor_commands(coord[1], 1),  "examples/quadrotor-lqr/figures/motor_commands_agent1.png")

println("Done. Plots written to examples/quadrotor-lqr/figures/ directory.")
