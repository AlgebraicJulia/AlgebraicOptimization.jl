module QuadrotorLQR

using LinearAlgebra
using Plots
using ControlSystems
using ..CellularSheaves
using ..VehicleInterface
using ..Controllers
import ..VehicleInterface: compute_control

export QuadrotorParams, DEFAULT_PARAMS
export QuadrotorModel
export LQRController
export PIDController
export formation_coboundary, SwarmCoordinator, sheaf_plan_step!, swarm_step!
export run_baseline_sim, run_coordinated_sim
export plot_trajectories, plot_formation_error, compare_runs

"""
    QuadrotorParams

Physical parameters for a quadrotor (SI units).

Dynamics derived from: Bouabdallah, "Design and control of quadrotors with
application to autonomous flying", EPFL PhD thesis, 2007.
Default values from: An et al., AIMS Electronics and Electrical
Engineering, doi:10.3934/electreng.2026002.
"""
struct QuadrotorParams
    m::Float64   # total mass (kg)
    g::Float64   # gravitational acceleration (m/s²)
    Ix::Float64  # roll moment of inertia (kg·m²)
    Iy::Float64  # pitch moment of inertia (kg·m²)
    Iz::Float64  # yaw moment of inertia (kg·m²)
    l::Float64   # arm length (m)
    Jr::Float64  # rotor inertia (kg·m²)
    kf::Float64  # thrust coefficient (N·s²/rad²)
    km::Float64  # drag coefficient (N·m·s²/rad²)
end

const DEFAULT_PARAMS = QuadrotorParams(
    0.468,
    9.81,
    4.856e-3,
    4.856e-3,
    8.801e-3,
    0.225,
    3.357e-5,
    2.980e-6,
    1.140e-7,
)

# ── Model ─────────────────────────────────────────────────────────────────────

"""
    hover_equilibrium(p) -> (x0, ω0)

12-state hover equilibrium (all zeros) and per-rotor speed ω₀ = √(mg / 4kf).
"""
function hover_equilibrium(p::QuadrotorParams)
    ω0 = sqrt(p.m * p.g / (4 * p.kf))
    return zeros(12), ω0
end

"""
    linearize_hover(p) -> (A_pos, B_pos, A_att, B_att, A_z, B_z)

Analytic linearization about hover giving three decoupled subsystems:
- Position  [x, y, ẋ, ẏ]       → [φ_cmd, θ_cmd]
- Attitude  [φ, θ, ψ, p, q, r] → [U₂, U₃, U₄]
- Altitude  [z, ż]              → δU₁

Ref: Bouabdallah 2007, §3.3.
"""
function linearize_hover(p::QuadrotorParams)
    g = p.g
    Ix, Iy, Iz = p.Ix, p.Iy, p.Iz
    l, Jr = p.l, p.Jr
    kf = p.kf

    ω0 = sqrt(p.m * g / (4 * kf))
    Ωr0 = 0.0

    A_pos = [0.0 0.0 1.0 0.0;
             0.0 0.0 0.0 1.0;
             0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 0.0]

    B_pos = [0.0 0.0;
             0.0 0.0;
             0.0 g;
             -g  0.0]

    a_pq = Jr / Ix * Ωr0
    a_qp = -Jr / Iy * Ωr0

    A_att = [0.0 0.0 0.0 1.0  0.0  0.0;
             0.0 0.0 0.0 0.0  1.0  0.0;
             0.0 0.0 0.0 0.0  0.0  1.0;
             0.0 0.0 0.0 0.0  a_pq 0.0;
             0.0 0.0 0.0 a_qp 0.0  0.0;
             0.0 0.0 0.0 0.0  0.0  0.0]

    B_att = [0.0   0.0   0.0;
             0.0   0.0   0.0;
             0.0   0.0   0.0;
             l/Ix  0.0   0.0;
             0.0   l/Iy  0.0;
             0.0   0.0   1/Iz]

    A_z = [0.0 1.0;
           0.0 0.0]

    B_z = reshape([0.0; 1.0/p.m], 2, 1)

    return A_pos, B_pos, A_att, B_att, A_z, B_z
end

"""
    QuadrotorModel <: AbstractVehicleModel

Full nonlinear quadrotor. Implements the VehicleInterface.

    State:   x = [x, y, z, ẋ, ẏ, ż, φ, θ, ψ, p, q, r]  (12-dim)
    Control: u = [U₁, U₂, U₃, U₄]  (thrust + attitude virtual inputs)

Ref: Bouabdallah 2007; Khan et al., "Robust Control of a Quadrotor",
IEEE Access, 2024.
"""
struct QuadrotorModel <: AbstractVehicleModel
    params::QuadrotorParams
end

QuadrotorModel() = QuadrotorModel(DEFAULT_PARAMS)

VehicleInterface.state_dim(::QuadrotorModel) = 12
VehicleInterface.control_dim(::QuadrotorModel) = 4
VehicleInterface.position_indices(::QuadrotorModel) = 1:3

function VehicleInterface.equilibrium(model::QuadrotorModel)
    x0, _ = hover_equilibrium(model.params)
    u0 = [model.params.m * model.params.g, 0.0, 0.0, 0.0]
    return x0, u0
end

"""
    dynamics(model::QuadrotorModel, x, u) -> ẋ

Full nonlinear quadrotor dynamics. Gyroscopic rotor coupling omitted (Ωr ≈ 0
at symmetric hover). ForwardDiff-compatible via `promote_type`.

Ref: Bouabdallah 2007, §4.2.
"""
function VehicleInterface.dynamics(model::QuadrotorModel, x::AbstractVector, u::AbstractVector)
    p = model.params
    m, g = p.m, p.g
    Ix, Iy, Iz, l = p.Ix, p.Iy, p.Iz, p.l

    φ, θ, ψ = x[7], x[8], x[9]
    pv, q, r = x[10], x[11], x[12]
    U1, U2, U3, U4 = u[1], u[2], u[3], u[4]

    cφ, sφ = cos(φ), sin(φ)
    cθ, sθ, tθ = cos(θ), sin(θ), tan(θ)
    cψ, sψ = cos(ψ), sin(ψ)

    T = promote_type(eltype(x), eltype(u))
    ẋ = Vector{T}(undef, 12)

    # Position kinematics
    ẋ[1] = x[4]
    ẋ[2] = x[5]
    ẋ[3] = x[6]

    # Translational dynamics (world frame)
    ẋ[4] = (cφ*sθ*cψ + sφ*sψ) * U1/m
    ẋ[5] = (cφ*sθ*sψ - sφ*cψ) * U1/m
    ẋ[6] = -g + cφ*cθ * U1/m

    # Euler angle kinematics
    ẋ[7] = pv + (q*sφ + r*cφ) * tθ
    ẋ[8] = q*cφ - r*sφ
    ẋ[9] = (q*sφ + r*cφ) / cθ

    # Rotational dynamics (body frame)
    ẋ[10] = (Iy - Iz)/Ix * q*r + l/Ix * U2
    ẋ[11] = (Iz - Ix)/Iy * pv*r + l/Iy * U3
    ẋ[12] = (Ix - Iy)/Iz * pv*q + 1/Iz * U4

    return ẋ
end

# ── LQR Controller ────────────────────────────────────────────────────────────

"""
    LQRController

Cascade LQR for the quadrotor. Solves three decoupled CAREs (position,
attitude, altitude) rather than one 12-state CARE, which is ill-conditioned.
Gains from Khan et al. 2024 are used as defaults.

Ref: Khan et al., "Development of an LQR-Based Control Algorithm
for Quadcopter", IEEE Access, 2024.
"""
struct LQRController
    K_pos::Matrix{Float64}
    K_att::Matrix{Float64}
    K_z::Matrix{Float64}
    params::QuadrotorParams
end

function LQRController(
    p::QuadrotorParams = DEFAULT_PARAMS;
    Q_pos::AbstractMatrix = Diagonal([0.1, 0.1, 0.001, 0.001]),
    R_pos::AbstractMatrix = Diagonal([1.0, 1.0]),
    Q_att::AbstractMatrix = Diagonal([20.0, 17.0, 0.15, 0.05, 0.05, 0.09]),
    R_att::AbstractMatrix = Diagonal([1.0, 1.0, 1.0]),
    Q_z::AbstractMatrix = Diagonal([0.03, 0.05]),
    R_z::AbstractMatrix = Diagonal([0.0002]),
)
    A_pos, B_pos, A_att, B_att, A_z, B_z = linearize_hover(p)

    for (name, A, B) in (("position", A_pos, B_pos), ("attitude", A_att, B_att), ("altitude", A_z, B_z))
        r = rank(ctrb(A, B))
        r == size(A, 1) || @warn "$name subsystem is NOT controllable (rank = $r)"
    end

    K_pos = lqr(A_pos, B_pos, Matrix(Q_pos), Matrix(R_pos))
    K_att = lqr(A_att, B_att, Matrix(Q_att), Matrix(R_att))
    K_z = lqr(A_z, B_z, Matrix(Q_z), Matrix(R_z))

    return LQRController(K_pos, K_att, K_z, p)
end

"""
    compute_control(ctrl::LQRController, x, x_ref) -> U

Cascade control law. Position loop computes attitude angle commands; attitude
loop computes torques. Returns U = [U₁, U₂, U₃, U₄].
"""
function compute_control(ctrl::LQRController, x::AbstractVector, x_ref::AbstractVector)
    p = ctrl.params

    e_pos = x[[1, 2, 4, 5]] - x_ref[[1, 2, 4, 5]]
    e_z = x[[3, 6]] - x_ref[[3, 6]]

    φθ_cmd = -ctrl.K_pos * e_pos
    δU1 = (-ctrl.K_z * e_z)[1]

    att_ref = [x_ref[7] + φθ_cmd[1], x_ref[8] + φθ_cmd[2], x_ref[9],
               x_ref[10], x_ref[11], x_ref[12]]
    U_att = -ctrl.K_att * (x[7:12] - att_ref)

    U1 = p.m * p.g + δU1
    U2, U3, U4 = U_att

    return [U1, U2, U3, U4]
end

# ── PID Controller ────────────────────────────────────────────────────────────

"""
    PIDController

Cascaded PID tuned via `loopshapingPID`. Each channel is a SISO loop over
its double-integrator plant P(s) = gain/s² at hover. Gains target a specified
crossover frequency.

State velocities and angular rates are used directly as derivative terms,
avoiding finite differences.
"""
mutable struct PIDController
    Kp_pos::Vector{Float64}  # [x, y]
    Ki_pos::Vector{Float64}
    Kd_pos::Vector{Float64}
    Kp_z::Float64
    Ki_z::Float64
    Kd_z::Float64
    Kp_att::Vector{Float64}  # [φ, θ, ψ]
    Ki_att::Vector{Float64}
    Kd_att::Vector{Float64}
    e_int_pos::Vector{Float64}
    e_int_z::Float64
    e_int_att::Vector{Float64}
    dt::Float64
    params::QuadrotorParams
end

function PIDController(
    p::QuadrotorParams = DEFAULT_PARAMS;
    ω_att::Float64 = 10.0,
    ω_pos::Float64 = 1.5,
    ω_z::Float64 = 2.0,
    dt::Float64 = 1e-3,
)
    g, m, l = p.g, p.m, p.l
    Ix, Iy, Iz = p.Ix, p.Iy, p.Iz

    # Each channel is a decoupled double integrator at hover: P(s) = gain/s²
    P_pos = tf(g, [1.0, 0.0, 0.0])
    P_z = tf(1/m, [1.0, 0.0, 0.0])
    P_φ = tf(l/Ix, [1.0, 0.0, 0.0])
    P_θ = tf(l/Iy, [1.0, 0.0, 0.0])
    P_ψ = tf(1/Iz, [1.0, 0.0, 0.0])

    _, kp_pos, ki_pos, kd_pos, _, _ = loopshapingPID(P_pos, ω_pos; doplot=false, form=:parallel)
    _, kp_z, ki_z, kd_z, _, _ = loopshapingPID(P_z, ω_z; doplot=false, form=:parallel)
    _, kp_φ, ki_φ, kd_φ, _, _ = loopshapingPID(P_φ, ω_att; doplot=false, form=:parallel)
    _, kp_θ, ki_θ, kd_θ, _, _ = loopshapingPID(P_θ, ω_att; doplot=false, form=:parallel)
    _, kp_ψ, ki_ψ, kd_ψ, _, _ = loopshapingPID(P_ψ, ω_att * 0.3; doplot=false, form=:parallel)

    # x and y share the same plant
    return PIDController(
        [kp_pos, kp_pos], [ki_pos, ki_pos], [kd_pos, kd_pos],
        kp_z, ki_z, kd_z,
        [kp_φ, kp_θ, kp_ψ], [ki_φ, ki_θ, ki_ψ], [kd_φ, kd_θ, kd_ψ],
        zeros(2), 0.0, zeros(3),
        dt, p,
    )
end

"""
    compute_control(ctrl::PIDController, x, x_ref) -> U

Cascaded PID control law. Position loop outputs angle commands which feed into
the attitude reference. Returns U = [U₁, U₂, U₃, U₄].
"""
function compute_control(ctrl::PIDController, x::AbstractVector, x_ref::AbstractVector)
    p = ctrl.params
    dt = ctrl.dt

    # Position loop — ẋ, ẏ are states 4, 5
    e_x = x[1] - x_ref[1]
    e_y = x[2] - x_ref[2]
    ė_x = x[4] - x_ref[4]
    ė_y = x[5] - x_ref[5]
    ctrl.e_int_pos .+= [e_x, e_y] * dt
    θ_cmd = -(ctrl.Kp_pos[1]*e_x + ctrl.Ki_pos[1]*ctrl.e_int_pos[1] + ctrl.Kd_pos[1]*ė_x)
    φ_cmd = -(ctrl.Kp_pos[2]*e_y + ctrl.Ki_pos[2]*ctrl.e_int_pos[2] + ctrl.Kd_pos[2]*ė_y)

    # Altitude loop — ż is state 6
    e_z = x[3] - x_ref[3]
    ė_z = x[6] - x_ref[6]
    ctrl.e_int_z += e_z * dt
    δU1 = -(ctrl.Kp_z*e_z + ctrl.Ki_z*ctrl.e_int_z + ctrl.Kd_z*ė_z)

    # Attitude loop — angle commands from position loop added to attitude reference
    e_ang = x[7:9] - [x_ref[7] + φ_cmd, x_ref[8] + θ_cmd, x_ref[9]]
    e_rate = x[10:12] - x_ref[10:12]
    ctrl.e_int_att .+= e_ang * dt
    U2 = -(ctrl.Kp_att[1]*e_ang[1] + ctrl.Ki_att[1]*ctrl.e_int_att[1] + ctrl.Kd_att[1]*e_rate[1])
    U3 = -(ctrl.Kp_att[2]*e_ang[2] + ctrl.Ki_att[2]*ctrl.e_int_att[2] + ctrl.Kd_att[2]*e_rate[2])
    U4 = -(ctrl.Kp_att[3]*e_ang[3] + ctrl.Ki_att[3]*ctrl.e_int_att[3] + ctrl.Kd_att[3]*e_rate[3])

    U1 = p.m * p.g + δU1

    return [U1, U2, U3, U4]
end

# ── Sheaf coordination ────────────────────────────────────────────────────────

"""
    formation_coboundary(n_agents, edges, offsets) -> (D, b)

Build coboundary matrix D ∈ ℝ^{3mx3n} and offset vector b for a position
formation sheaf over a graph with n agents and m edges. Formation is satisfied
when Dx = b.

Ref: Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves",
Journal of Applied and Computational Topology, 2019.
"""
function formation_coboundary(
    n_agents::Int,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}} = Dict{Tuple{Int,Int}, Vector{Float64}}(),
)
    sheaf, b = _formation_sheaf(n_agents, edges, offsets)
    D = Matrix{Float64}(coboundary_map(sheaf))
    return D, b
end

function _formation_sheaf(
    n_agents::Int,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}},
)
    n_edges = length(edges)
    sheaf = CellularSheaf(fill(3, n_agents), fill(3, n_edges))
    b = zeros(3 * n_edges)
    I3 = Matrix(1.0 * I, 3, 3)
    for (e, (v1, v2)) in enumerate(edges)
        set_edge_maps!(sheaf, v1, v2, e, I3, I3)
        b[3*(e-1)+1:3*e] = get(offsets, (v1, v2), zeros(3))
    end
    return sheaf, b
end

"""
    SwarmCoordinator

N-agent swarm using a cellular sheaf Laplacian for formation coordination.
Slow outer planner sets position references via `sheaf_plan_step!`;
fast inner loop tracks them via `swarm_step!`.
"""
mutable struct SwarmCoordinator
    sheaf::CellularSheaf
    b::Vector{Float64}
    controllers::Vector{SheafControllerInterface}
    n_agents::Int
end

function SwarmCoordinator(
    n_agents::Int,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}} = Dict{Tuple{Int,Int}, Vector{Float64}}();
    params::QuadrotorParams = DEFAULT_PARAMS,
    ctrl = LQRController(params),
    planner_hz::Float64 = 10.0,
)
    sheaf, b = _formation_sheaf(n_agents, edges, offsets)
    controllers = [SheafControllerInterface(deepcopy(ctrl); planner_hz=planner_hz)
                   for _ in 1:n_agents]
    return SwarmCoordinator(sheaf, b, controllers, n_agents)
end

"""
    sheaf_plan_step!(coord, states, t_now) -> pos_new

Outer planner tick. Projects current positions onto the formation-consistent
subspace {x : Dx = b} and updates each agent's reference.
"""
function sheaf_plan_step!(coord::SwarmCoordinator, states::Vector{<:AbstractVector}, t_now::Float64)
    @assert length(states) == coord.n_agents "expected $(coord.n_agents) state vectors"

    pos = vcat([s[1:3] for s in states]...)
    pos_new = Vector{Float64}(nearest_section(coord.sheaf, pos, coord.b))

    for i in 1:coord.n_agents
        x_ref_new = copy(coord.controllers[i].x_ref)
        x_ref_new[1:3] .= pos_new[3*(i-1)+1:3*i]
        update_reference!(coord.controllers[i], x_ref_new, t_now)
    end

    return pos_new
end

"""
    swarm_step!(coord, states) -> Vector{Vector}

Inner-loop tick: evaluate `compute_control` for each agent.
"""
function swarm_step!(coord::SwarmCoordinator, states::Vector{<:AbstractVector})
    @assert length(states) == coord.n_agents "expected $(coord.n_agents) state vectors"
    return [step!(coord.controllers[i], states[i]) for i in 1:coord.n_agents]
end

# ── Simulation ────────────────────────────────────────────────────────────────

"""
    run_baseline_sim(waypoints; params, ctrl, x0s, dt, t_end) -> Vector{SimRecord}

Each agent independently tracks its own fixed waypoint with no inter-agent coupling.
"""
function run_baseline_sim(
    waypoints::Vector{<:AbstractVector};
    params::QuadrotorParams = DEFAULT_PARAMS,
    ctrl = LQRController(params),
    x0s::Vector{<:AbstractVector} = [zeros(12) for _ in eachindex(waypoints)],
    dt::Float64 = 1e-3,
    t_end::Float64 = 10.0,
    planner_hz::Float64 = 10.0,
)
    n_agents = length(waypoints)
    n_steps = round(Int, t_end / dt)
    model = LinearizedModel(QuadrotorModel(params))
    agent_ctrls = [deepcopy(ctrl) for _ in 1:n_agents]

    records = [SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 4, n_steps),
    ) for _ in 1:n_agents]

    states = [copy(x0s[i]) for i in 1:n_agents]

    for k in 1:n_steps
        t_now = (k - 1) * dt
        for i in 1:n_agents
            x = states[i]
            x_ref = waypoints[i]
            U = compute_control(agent_ctrls[i], x, x_ref)
            records[i].t[k] = t_now
            records[i].x[:, k] = x
            records[i].x_ref[:, k] = x_ref
            records[i].u[:, k] = U
            states[i] = _rk4(model, x, U, dt)
        end
    end

    return records
end

"""
    run_coordinated_sim(n_agents, edges, offsets; ...) -> Vector{SimRecord}

Sheaf-coordinated simulation. Outer planner updates position references at
`planner_hz`; each agent's inner loop runs at 1/dt.
"""
function run_coordinated_sim(
    n_agents::Int,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}} = Dict{Tuple{Int,Int}, Vector{Float64}}();
    params::QuadrotorParams = DEFAULT_PARAMS,
    ctrl = LQRController(params),
    x0s::Vector{<:AbstractVector} = [zeros(12) for _ in 1:n_agents],
    dt::Float64 = 1e-3,
    t_end::Float64 = 10.0,
    planner_hz::Float64 = 10.0,
)
    n_steps = round(Int, t_end / dt)
    planner_ticks = round(Int, (1.0 / planner_hz) / dt)
    model = LinearizedModel(QuadrotorModel(params))

    coord = SwarmCoordinator(n_agents, edges, offsets; params=params, ctrl=ctrl, planner_hz=planner_hz)

    for i in 1:n_agents
        update_reference!(coord.controllers[i], copy(x0s[i]), 0.0)
    end

    records = [SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 4, n_steps),
    ) for _ in 1:n_agents]

    states = [copy(x0s[i]) for i in 1:n_agents]

    for k in 1:n_steps
        t_now = (k - 1) * dt

        if mod(k - 1, planner_ticks) == 0
            sheaf_plan_step!(coord, states, t_now)
        end

        results = swarm_step!(coord, states)

        for i in 1:n_agents
            U = results[i]
            records[i].t[k] = t_now
            records[i].x[:, k] = states[i]
            records[i].x_ref[:, k] = coord.controllers[i].x_ref
            records[i].u[:, k] = U
            states[i] = _rk4(model, states[i], U, dt)
        end
    end

    return records
end

# ── Plotting ──────────────────────────────────────────────────────────────────

"""
    plot_trajectories(records; title) -> Plot

x, y, z time series for all agents. Dashed lines show references.
"""
function plot_trajectories(records::Vector{SimRecord}; title::String = "Position trajectories")
    t = records[1].t
    cols = palette(:tab10)
    labels = ("x [m]", "y [m]", "z [m]")

    p = plot(layout=(3, 1), size=(800, 600), link=:x, plot_title=title,
             left_margin=10Plots.mm, bottom_margin=6Plots.mm)

    for (row, (lbl, idx)) in enumerate(zip(labels, (1, 2, 3)))
        for (j, rec) in enumerate(records)
            plot!(p[row], t, rec.x[idx, :];
                label=row == 1 ? "Agent $j" : "",
                color=cols[j],
                ylabel=lbl,
                xlabel=row == 3 ? "t [s]" : "",
                legend=row == 1 ? :topright : false,
                linewidth=1.5)
            plot!(p[row], t, rec.x_ref[idx, :];
                label="",
                color=cols[j],
                linestyle=:dash,
                alpha=0.5,
                linewidth=1.0)
        end
    end
    return p
end

"""
    plot_formation_error(records, D, b; title) -> Plot

Formation error ‖Dx - b‖ over time.
"""
function plot_formation_error(
    records::Vector{SimRecord},
    D::Matrix{Float64},
    b::Vector{Float64};
    title::String = "Formation consistency error",
)
    t = records[1].t
    err = [norm(D * vcat([records[i].x[1:3, k] for i in eachindex(records)]...) - b)
           for k in 1:length(t)]

    return plot(t, err;
        xlabel="t [s]",
        ylabel="‖Dx - b‖  [m]",
        title=title,
        legend=false,
        linewidth=2,
        color=:steelblue,
        size=(800, 300),
        left_margin=10Plots.mm,
        bottom_margin=8Plots.mm)
end

"""
    compare_runs(baseline, coordinated, D, b; title, label1, label2) -> Plot

2x2 comparison: z-trajectories and formation error for two runs.
"""
function compare_runs(
    baseline::Vector{SimRecord},
    coordinated::Vector{SimRecord},
    D::Matrix{Float64},
    b::Vector{Float64};
    title::String = "Baseline vs. Sheaf-Coordinated",
    label1::String = "Baseline",
    label2::String = "Coordinated",
)
    t = baseline[1].t
    cols = palette(:tab10)
    n_agents = length(baseline)

    function _err(recs)
        [norm(D * vcat([recs[i].x[1:3, k] for i in 1:n_agents]...) - b)
         for k in eachindex(recs[1].t)]
    end

    margin = 8Plots.mm
    pz_base = plot(title="$label1 — z [m]", xlabel="", ylabel="z [m]",
                   legend=:bottomright, left_margin=margin)
    pz_coord = plot(title="$label2 — z [m]", xlabel="", ylabel="z [m]",
                    legend=:bottomright, left_margin=margin)
    pe_base = plot(title="$label1 — formation error", xlabel="t [s]", ylabel="‖Dx-b‖ [m]",
                   legend=false, color=:crimson, linewidth=2,
                   left_margin=margin, bottom_margin=margin)
    pe_coord = plot(title="$label2 — formation error", xlabel="t [s]", ylabel="‖Dx-b‖ [m]",
                    legend=false, color=:steelblue, linewidth=2,
                    left_margin=margin, bottom_margin=margin)

    for j in 1:n_agents
        plot!(pz_base, t, baseline[j].x[3, :]; color=cols[j], label="Agent $j", lw=1.5)
        plot!(pz_coord, t, coordinated[j].x[3, :]; color=cols[j], label="Agent $j", lw=1.5)
    end

    plot!(pe_base, t, _err(baseline); color=:crimson)
    plot!(pe_coord, t, _err(coordinated); color=:steelblue)

    return plot(pz_base, pz_coord, pe_base, pe_coord;
                layout=(2, 2), size=(1000, 600), plot_title=title)
end

end # module
