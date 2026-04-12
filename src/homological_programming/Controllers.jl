module Controllers

using LinearAlgebra
using ControlSystems
using ..VehicleInterface
import ..VehicleInterface: compute_control

export LinearLQRController
export SheafControllerInterface, update_reference!, step!
export SimRecord, _rk4

"""
    SimRecord

Generic per-agent simulation history. Vehicle-agnostic.

    t     — time vector (n_steps,)
    x     — state history (n_states x n_steps)
    x_ref — reference history (n_states x n_steps)
    u     — control input history (n_controls x n_steps)
"""
struct SimRecord
    t::Vector{Float64}
    x::Matrix{Float64}
    x_ref::Matrix{Float64}
    u::Matrix{Float64}
end

"""
    _rk4(model, x, u, dt) -> x_next

RK4 integration step through `dynamics(model, x, u)`.
All vehicle simulations use this as their stepping primitive.
"""
function _rk4(model::AbstractVehicleModel, x::AbstractVector, u::AbstractVector, dt::Float64)
    f(s) = dynamics(model, s, u)
    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt * k3)
    return x + (dt/6) .* (k1 + 2k2 + 2k3 + k4)
end

"""
    LinearLQRController

Full-state LQR for any `AbstractVehicleModel`. Linearizes the model at its
equilibrium via ForwardDiff and solves the CARE for gain K.

    u = u_eq - K·(x - x_ref)
"""
struct LinearLQRController
    K::Matrix{Float64}
    x_eq::Vector{Float64}
    u_eq::Vector{Float64}
end

function LinearLQRController(
    model::AbstractVehicleModel;
    Q::AbstractMatrix = Matrix(1.0 * I(state_dim(model))),
    R::AbstractMatrix = Matrix(1.0 * I(control_dim(model))),
)
    A, B = linearize(model)
    x_eq, u_eq = equilibrium(model)

    r = rank(ctrb(A, B))
    r == size(A, 1) || @warn "$(typeof(model)) is not fully controllable (rank = $r)"

    K = lqr(A, B, Matrix(Q), Matrix(R))
    return LinearLQRController(K, x_eq, u_eq)
end

function compute_control(ctrl::LinearLQRController, x::AbstractVector, x_ref::AbstractVector)
    return ctrl.u_eq .- ctrl.K * (x .- x_ref)
end

"""
    SheafControllerInterface

Mediates time-scale separation between the slow sheaf planner and the fast
inner control loop. Controller-agnostic: works with any type implementing
`compute_control(ctrl, x, x_ref)`.
"""
mutable struct SheafControllerInterface
    ctrl
    x_ref::Vector{Float64}
    t_last_plan::Float64
    planner_dt::Float64
end

function SheafControllerInterface(ctrl; n_states::Int, planner_hz::Float64 = 10.0)
    return SheafControllerInterface(ctrl, zeros(n_states), -Inf, 1.0 / planner_hz)
end

function SheafControllerInterface(ctrl, model::AbstractVehicleModel; planner_hz::Float64 = 10.0)
    return SheafControllerInterface(ctrl; n_states=state_dim(model), planner_hz=planner_hz)
end

"""
    update_reference!(iface, x_ref_new, t_now)

Write a new reference from the planner into the interface.
"""
function update_reference!(iface::SheafControllerInterface, x_ref_new::AbstractVector, t_now::Float64)
    @assert length(x_ref_new) == length(iface.x_ref) "x_ref length mismatch: " *
        "expected $(length(iface.x_ref)), got $(length(x_ref_new))"
    copyto!(iface.x_ref, x_ref_new)
    iface.t_last_plan = t_now
    return iface
end

"""
    step!(iface, x) -> u

Inner-loop tick: evaluate `compute_control(ctrl, x, x_ref)`.
"""
function step!(iface::SheafControllerInterface, x::AbstractVector)
    return compute_control(iface.ctrl, x, iface.x_ref)
end

end
