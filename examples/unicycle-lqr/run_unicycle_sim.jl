using AlgebraicOptimization
using Plots
using LinearAlgebra
using Dates

v0    = 4.0   # forward speed (m/s) — high enough to make coupling visible
model = UnicycleModel(v0)

Q = Diagonal([1.0, 0.1])       # penalise lateral error more than heading
R = reshape([1.0], 1, 1)

println("Running LQR simulation...")
lqr_ctrl = LinearLQRController(model; Q=Q, R=R)
lqr_run  = run_unicycle_sim(model, lqr_ctrl; x0=[1.0, 0.5], t_end=10.0)

println("Running PID simulation...")
pid_ctrl = UnicyclePIDController(model; ω_c=1.5)
pid_run  = run_unicycle_sim(model, pid_ctrl; x0=[1.0, 0.5], t_end=10.0)

println("\nFinal errors at t = 10 s:")
println("  LQR — e_y: ", round(lqr_run.x[1, end], sigdigits=4),
        " m   e_ψ: ", round(rad2deg(lqr_run.x[2, end]), sigdigits=4), "°")
println("  PID — e_y: ", round(pid_run.x[1, end], sigdigits=4),
        " m   e_ψ: ", round(rad2deg(pid_run.x[2, end]), sigdigits=4), "°")

println("\nPeak lateral error:")
println("  LQR: ", round(maximum(abs.(lqr_run.x[1, :])), sigdigits=4), " m")
println("  PID: ", round(maximum(abs.(pid_run.x[1, :])), sigdigits=4), " m")

println("\nSaving plots...")
fp   = "examples/unicycle-lqr/figures/"
isdir(fp) || mkdir(fp)
date = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

savefig(compare_unicycle_runs(pid_run, lqr_run;
        label1="PID", label2="LQR",
        title="Coordinated PID vs. LQR (v₀ = $(v0) m/s)"),
        fp * "comparison_" * date * ".png")

savefig(plot_unicycle_tracking(lqr_run; title="LQR tracking (v₀ = $(v0) m/s)"),
        fp * "tracking_lqr_" * date * ".png")

savefig(plot_unicycle_tracking(pid_run; title="PID tracking (v₀ = $(v0) m/s)"),
        fp * "tracking_pid_" * date * ".png")

println("Done. Plots written to " * fp)
