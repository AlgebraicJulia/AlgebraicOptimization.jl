module SubmarineLQR

using LinearAlgebra
using ..VehicleInterface
using ..Controllers

export SubmarineParams, DEFAULT_SUBMARINE_PARAMS
export SubmarineModel

# ── Model ─────────────────────────────────────────────────────────────────────

"""
    SubmarineParams

Physical parameters for a neutrally buoyant AUV (SI units).

Buoyancy is trimmed to exactly cancel gravity, so the equilibrium input is
zero. Added mass is modelled as a diagonal tensor — valid for a body of
revolution (torpedo shape). Drag is quadratic in velocity (Xuu·u|u| form).

The surge axis (x) is streamlined: low added mass, low drag. The sway (y) and
heave (z) axes are blunt: large added mass, high drag. This gives distinct
effective inertias and response times across axes.

Linearized at zero velocity, quadratic drag vanishes (d/dv(v|v|)|₀ = 0), so
the linearized model is three decoupled double integrators.

Default values from Ridley, Fontan & Corke, "Submarine Dynamic Modelling",
ARCA 2003, Table 1.

Note: the paper lists Yvv = 3.01 kg/m and Zww = 30.1 kg/m; the 10x difference
is inconsistent with a body of revolution and may be a transcription error. We
assume the same drag in sway and heave, consistent with the added mass values and a
torpedo-shaped body.

Ref: Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion Control",
Wiley, 2011, §6.3 (sign convention and added mass tensor).
"""
struct SubmarineParams
    m::Float64     # dry mass (kg)
    mx::Float64    # added mass, surge (kg)   — Ẋu
    my::Float64    # added mass, sway  (kg)   — Ẏv
    mz::Float64    # added mass, heave (kg)   — Żw
    dxx::Float64   # quadratic drag, surge (kg/m)  — |Xuu|
    dyy::Float64   # quadratic drag, sway  (kg/m)  — |Yvv|
    dzz::Float64   # quadratic drag, heave (kg/m)  — |Zww|
end

const DEFAULT_SUBMARINE_PARAMS = SubmarineParams(
    18.826,   # m   (kg)
    0.421,    # mx  (kg)   — Ẋu
    27.2,     # my  (kg)   — |Ẏv|
    27.2,     # mz  (kg)   — |Żw|
    3.11,     # dxx (kg/m) — |Xuu|
    30.1,     # dyy (kg/m) — |Yvv|
    30.1,     # dzz (kg/m) — |Zww|
)

# ── SubmarineModel ────────────────────────────────────────────────────────────

"""
    SubmarineModel <: AbstractVehicleModel

Neutrally buoyant AUV in 3D. Implements the VehicleInterface.

    State:   x = [x, y, z, ẋ, ẏ, ż]  (6-dim)
    Control: u = [Fx, Fy, Fz]        (body-frame thruster forces, N)

Buoyancy cancels gravity, so the nominal input is zero. Drag is quadratic:

    (m + mx)ẍ = Fx - dxx·ẋ|ẋ|
    (m + my)ÿ = Fy - dyy·ẏ|ẏ|
    (m + mz)z̈ = Fz - dzz·ż|ż|

At zero velocity the linearized drag is zero, so `LinearizedModel` produces
three decoupled double integrators. LQR provides all active damping and
position control.

Ref: Ridley, Fontan & Corke, "Submarine Dynamic Modelling", ARCA 2003.
"""
struct SubmarineModel <: AbstractVehicleModel
    params::SubmarineParams
end

SubmarineModel() = SubmarineModel(DEFAULT_SUBMARINE_PARAMS)

VehicleInterface.state_dim(::SubmarineModel) = 6
VehicleInterface.control_dim(::SubmarineModel) = 3
VehicleInterface.position_indices(::SubmarineModel) = 1:3

function VehicleInterface.equilibrium(::SubmarineModel)
    return zeros(6), zeros(3)
end

"""
    dynamics(model::SubmarineModel, x, u) -> ẋ

Translational dynamics with diagonal added mass and quadratic drag. Buoyancy
exactly cancels gravity. ForwardDiff-compatible via `promote_type`.
"""
function VehicleInterface.dynamics(model::SubmarineModel, x::AbstractVector, u::AbstractVector)
    p = model.params
    T = promote_type(eltype(x), eltype(u))
    ẋ = Vector{T}(undef, 6)

    mx_eff = p.m + p.mx
    my_eff = p.m + p.my
    mz_eff = p.m + p.mz

    # Position kinematics
    ẋ[1] = x[4]
    ẋ[2] = x[5]
    ẋ[3] = x[6]

    # Translational dynamics: thrust minus quadratic drag, divided by effective mass
    ẋ[4] = u[1] / mx_eff - (p.dxx / mx_eff) * x[4] * abs(x[4])
    ẋ[5] = u[2] / my_eff - (p.dyy / my_eff) * x[5] * abs(x[5])
    ẋ[6] = u[3] / mz_eff - (p.dzz / mz_eff) * x[6] * abs(x[6])

    return ẋ
end

end
