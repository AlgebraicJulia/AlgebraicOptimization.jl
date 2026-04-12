using AlgebraicOptimization
using Plots
using LinearAlgebra
using Dates

edges   = [(1, 2), (2, 3)]
offsets = Dict((1, 2) => [2.0, 0.0, 0.0],
               (2, 3) => [2.0, 0.0, 0.0])

x0s = [zeros(12) for _ in 1:3]
D, b = formation_coboundary(3, edges, offsets)

println("Running coordinated LQR simulation...")
lqr_runs = run_coordinated_sim(3, edges, offsets; ctrl=LQRController(), x0s=x0s, t_end=10.0)

println("Running coordinated PID simulation...")
pid_runs = run_coordinated_sim(3, edges, offsets; ctrl=PIDController(), x0s=x0s, t_end=10.0)

err(recs) = norm(D * vcat([recs[i].x[1:3, end] for i in 1:3]...) - b)
println("\nFormation error at t = 10 s:")
println("  LQR: ", round(err(lqr_runs), sigdigits=4), " m")
println("  PID: ", round(err(pid_runs), sigdigits=4), " m")

println("\nSaving plots...")
fp = "examples/quadrotor-lqr/figures/"
isdir(fp) || mkdir(fp)
date = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

savefig(compare_runs(pid_runs, lqr_runs, D, b; title="Coordinated PID vs. Coordinated LQR", label1="PID", label2="LQR"), fp * "comparison_" * date * ".png")
savefig(plot_trajectories(lqr_runs; title="LQR trajectories"), fp * "trajectories_lqr_" * date * ".png")
savefig(plot_trajectories(pid_runs; title="PID trajectories"), fp * "trajectories_pid_" * date * ".png")
savefig(plot_formation_error(lqr_runs, D, b; title="LQR formation error"), fp * "formation_error_lqr_" * date * ".png")
savefig(plot_formation_error(pid_runs, D, b; title="PID formation error"), fp * "formation_error_pid_" * date * ".png")

println("Done. Plots written to " * fp)
