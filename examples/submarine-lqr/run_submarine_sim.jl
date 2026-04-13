using AlgebraicOptimization
using Plots
using LinearAlgebra
using Dates

edges = [(1, 2), (2, 3)]
offsets = Dict((1, 2) => [2.0, 0.0, 0.0],
               (2, 3) => [2.0, 0.0, 0.0])

n_agents = 3
D, b = formation_coboundary(n_agents, edges, offsets)

models = [SubmarineModel() for _ in 1:n_agents]
ctrls = [LinearLQRController(SubmarineModel()) for _ in 1:n_agents]

x0s = [zeros(6) for _ in 1:n_agents]

println("Running coordinated simulation (sheaf-projected)...")
coordinated_runs = run_coordinated_sim(models, ctrls, edges, offsets; x0s=x0s, t_end=120.0)

err(recs) = norm(D * vcat([recs[i].x[1:3, end] for i in 1:n_agents]...) - b)

println("\nFormation error at t = 120 s:")
println("  Coordinated: ", round(err(coordinated_runs), sigdigits=4), " m")

println("\nSaving plots...")
fp = "examples/submarine-lqr/figures/"
isdir(fp) || mkdir(fp)
date = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

savefig(
    plot_trajectories(coordinated_runs; title="Submarine LQR — trajectories"),
    fp * "trajectories_coordinated_" * date * ".png",
)

savefig(
    plot_formation_error(coordinated_runs, D, b; title="Submarine LQR — formation error"),
    fp * "formation_error_coordinated_" * date * ".png",
)

println("Done. Plots written to " * fp)
