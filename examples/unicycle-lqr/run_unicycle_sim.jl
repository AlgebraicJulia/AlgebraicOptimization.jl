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

pp_orange = RGB(223/255, 167/255, 119/255)
pp_blue   = RGB(97/255,  136/255, 178/255)
base_kw   = (fontfamily="Computer Modern", thickness_scaling=1.5,
             guidefontsize=10, tickfontsize=9,
             left_margin=8Plots.mm, bottom_margin=5Plots.mm,
             top_margin=8Plots.mm, size=(800, 300))

# Lateral error comparison
p_ey = plot(pid_run.t, pid_run.x[1, :];
            label="PID", color=pp_orange, linewidth=2.0,
            ylabel="Lateral error e_y (m)", xlabel="Time (s)", base_kw...)
plot!(p_ey, lqr_run.t, lqr_run.x[1, :]; label="LQR", color=pp_blue, linewidth=2.0)
hline!(p_ey, [0.0]; linestyle=:dash, color=:gray60, label="")
savefig(p_ey, fp * "comparison_ey_" * date * ".png")

# Heading error comparison
p_psi = plot(pid_run.t, rad2deg.(pid_run.x[2, :]);
             label="PID", color=pp_orange, linewidth=2.0,
             ylabel="Heading error e_psi (deg)", xlabel="Time (s)", base_kw...)
plot!(p_psi, lqr_run.t, rad2deg.(lqr_run.x[2, :]); label="LQR", color=pp_blue, linewidth=2.0)
hline!(p_psi, [0.0]; linestyle=:dash, color=:gray60, label="")
savefig(p_psi, fp * "comparison_psi_" * date * ".png")

# Yaw rate comparison
p_w = plot(pid_run.t, pid_run.u[1, :];
           label="PID", color=pp_orange, linewidth=2.0,
           ylabel="Yaw rate (rad/s)", xlabel="Time (s)", base_kw...)
plot!(p_w, lqr_run.t, lqr_run.u[1, :]; label="LQR", color=pp_blue, linewidth=2.0)
savefig(p_w, fp * "comparison_yawrate_" * date * ".png")

savefig(plot_unicycle_tracking(lqr_run; title="LQR tracking (v_0 = $(v0) m/s)"),
        fp * "tracking_lqr_" * date * ".png")

savefig(plot_unicycle_tracking(pid_run; title="PID tracking (v_0 = $(v0) m/s)"),
        fp * "tracking_pid_" * date * ".png")

println("Done. Plots written to " * fp)
