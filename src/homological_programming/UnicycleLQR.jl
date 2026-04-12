module UnicycleLQR

using LinearAlgebra
using Plots
using ControlSystems
using ..VehicleInterface
using ..Controllers
import ..VehicleInterface: compute_control

export UnicycleParams, DEFAULT_UNICYCLE_PARAMS
export UnicycleModel
export UnicyclePIDController
export run_unicycle_sim
export plot_unicycle_tracking, compare_unicycle_runs

# ── Model ─────────────────────────────────────────────────────────────────────

"""
    UnicycleParams

Parameters for the kinematic unicycle.
"""
struct UnicycleParams
    v0::Float64  # forward speed (m/s)
end

const DEFAULT_UNICYCLE_PARAMS = UnicycleParams(2.0)

"""
    UnicycleModel <: AbstractVehicleModel

Kinematic unicycle in path-tracking error coordinates.

    State:   x = [e_y, e_ψ]  — lateral error (m), heading error (rad)
    Control: u = [ω]         — yaw rate (rad/s)

Dynamics:  ė_y = v₀·sin(e_ψ),  ė_ψ = ω

The linearized A matrix has A[1,2] = v₀. A heading error drives lateral drift
at rate v₀, creating coupling that LQR exploits but a scalar PID on e_y alone
cannot fully handle. At v₀ = 0 the plant is decoupled.

Ref: Siegwart et al., "Introduction to Autonomous Mobile Robots", 2nd ed.,
MIT Press, 2011, §3.2.
"""
struct UnicycleModel <: AbstractVehicleModel
    params::UnicycleParams
end

UnicycleModel() = UnicycleModel(DEFAULT_UNICYCLE_PARAMS)
UnicycleModel(v0::Float64) = UnicycleModel(UnicycleParams(v0))

VehicleInterface.state_dim(::UnicycleModel) = 2
VehicleInterface.control_dim(::UnicycleModel) = 1
VehicleInterface.position_indices(::UnicycleModel) = 1:1

function VehicleInterface.equilibrium(::UnicycleModel)
    return zeros(2), zeros(1)
end

function VehicleInterface.dynamics(model::UnicycleModel, x::AbstractVector, u::AbstractVector)
    v0 = model.params.v0
    T = promote_type(eltype(x), eltype(u))
    ẋ = Vector{T}(undef, 2) 
    ẋ[1] = v0 * sin(x[2])
    ẋ[2] = u[1]
    return ẋ
end

# ── PID Controller ─────────────────────────────────────────────────────────────

"""
    UnicyclePIDController

PID for lateral path-tracking. The plant seen by the PID is P(s) = v₀/s²,
a double integrator. Gains are tuned via `loopshapingPID` at crossover
frequency ω_c.

The derivative term uses ė_y = v₀·sin(e_ψ) rather than a finite difference,
equivalent to a cascade PD with heading as the derivative state.
"""
mutable struct UnicyclePIDController
    Kp::Float64
    Ki::Float64
    Kd::Float64
    v0::Float64
    e_int::Float64
    dt::Float64
end

function UnicyclePIDController(model::UnicycleModel; ω_c::Float64 = 1.5, dt::Float64 = 1e-3)
    v0 = model.params.v0
    P = tf(v0, [1.0, 0.0, 0.0])
    _, kp, ki, kd, _, _ = loopshapingPID(P, ω_c; doplot=false, form=:parallel)
    return UnicyclePIDController(kp, ki, kd, v0, 0.0, dt)
end

function compute_control(ctrl::UnicyclePIDController, x::AbstractVector, x_ref::AbstractVector)
    dt = ctrl.dt
    e_y = x[1] - x_ref[1]
    ė_y = ctrl.v0 * sin(x[2] - x_ref[2])
    ctrl.e_int += e_y * dt
    ω = -(ctrl.Kp * e_y + ctrl.Ki * ctrl.e_int + ctrl.Kd * ė_y)
    return [ω]
end

# ── Simulation ─────────────────────────────────────────────────────────────────

const ω_MAX = 5.0  # rad/s — physical yaw rate limit

"""
    run_unicycle_sim(model, ctrl; x0, x_ref, dt, t_end) -> SimRecord

Simulate unicycle path tracking from initial state x0.
"""
function run_unicycle_sim(
    model::UnicycleModel,
    ctrl;
    x0::AbstractVector = [1.0, 0.3],
    x_ref::AbstractVector = zeros(2),
    dt::Float64 = 1e-3,
    t_end::Float64 = 10.0,
)
    n_steps = round(Int, t_end / dt)
    model = LinearizedModel(model)
    rec = SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, 2, n_steps),
        Matrix{Float64}(undef, 2, n_steps),
        Matrix{Float64}(undef, 1, n_steps),
    )
    x = copy(Vector{Float64}(x0))
    for k in 1:n_steps
        u = compute_control(ctrl, x, x_ref)
        u = clamp.(u, -ω_MAX, ω_MAX)
        rec.t[k] = (k - 1) * dt
        rec.x[:, k] = x
        rec.x_ref[:, k] = x_ref
        rec.u[:, k] = u
        x = _rk4(model, x, u, dt)
    end
    return rec
end

# ── Plotting ───────────────────────────────────────────────────────────────────

"""
    plot_unicycle_tracking(rec; title) -> Plot

Three-panel: lateral error, heading error, yaw rate vs time.
"""
function plot_unicycle_tracking(rec::SimRecord; title::String = "Unicycle Path Tracking")
    p1 = plot(rec.t, rec.x[1, :];
        label="e_y (m)", ylabel="Lateral error (m)",
        left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    hline!(p1, [0.0]; linestyle=:dash, color=:black, label="")

    p2 = plot(rec.t, rad2deg.(rec.x[2, :]);
        label="e_ψ (°)", ylabel="Heading error (°)",
        left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    hline!(p2, [0.0]; linestyle=:dash, color=:black, label="")

    p3 = plot(rec.t, rec.u[1, :];
        label="ω (rad/s)", ylabel="Yaw rate (rad/s)", xlabel="Time (s)",
        left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    return plot(p1, p2, p3; layout=(3, 1), plot_title=title, size=(800, 600))
end

"""
    compare_unicycle_runs(rec1, rec2; label1, label2, title) -> Plot

Overlay two runs on lateral error, heading error, and yaw rate.
"""
function compare_unicycle_runs(
    rec1::SimRecord,
    rec2::SimRecord;
    label1::String = "PID",
    label2::String = "LQR",
    title::String = "PID vs LQR",
)
    p1 = plot(rec1.t, rec1.x[1, :]; label=label1, ylabel="Lateral error e_y (m)",
              left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    plot!(p1, rec2.t, rec2.x[1, :]; label=label2)
    hline!(p1, [0.0]; linestyle=:dash, color=:black, label="")

    p2 = plot(rec1.t, rad2deg.(rec1.x[2, :]); label=label1, ylabel="Heading error e_ψ (°)",
              left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    plot!(p2, rec2.t, rad2deg.(rec2.x[2, :]); label=label2)
    hline!(p2, [0.0]; linestyle=:dash, color=:black, label="")

    p3 = plot(rec1.t, rec1.u[1, :]; label=label1, ylabel="Yaw rate ω (rad/s)",
              xlabel="Time (s)", left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    plot!(p3, rec2.t, rec2.u[1, :]; label=label2)

    return plot(p1, p2, p3; layout=(3, 1), plot_title=title, size=(800, 600))
end

end
