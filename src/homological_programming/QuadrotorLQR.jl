module QuadrotorLQR

using LinearAlgebra
using Plots
using ControlSystems
using ..CellularSheaves

export QuadrotorParams, DEFAULT_PARAMS
export hover_equilibrium, linearize_hover, assemble_full_plant
export LQRController
export compute_control, SheafControllerInterface, update_reference!, step!
export PIDController
export formation_coboundary, SwarmCoordinator, sheaf_plan_step!, swarm_step!
export SimRecord, run_baseline_sim, run_coordinated_sim
export plot_trajectories, plot_formation_error, plot_motor_commands, compare_runs

"""
    QuadrotorParams

Physical parameters (SI units).
"""
struct QuadrotorParams
    m::Float64  # total mass
    g::Float64  # gravity
    Ix::Float64 # roll moment of intertia
    Iy::Float64 # pitch moment of inertia
    Iz::Float64 # yaw moment of inertia
    l::Float64  # arm length
    Jr::Float64 # rotor inertia
    kf::Float64 # thrust coefficient
    km::Float64 # drag coefficient
end

# Symmetric test vehicle from Bouabdallah 2004
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

"""
    hover_equilibrium(p) -> (x0, ω0)

Returns the 12-state hover equilibrium (all zeros) and per-rotor speed ω₀ = √(mg/4kf).
"""
function hover_equilibrium(p::QuadrotorParams)
    ω0 = sqrt(p.m * p.g / (4 * p.kf))
    x0 = zeros(12)
    return x0, ω0
end

"""
    linearize_hover(p) -> (A_pos, B_pos, A_att, B_att, A_z, B_z)

Linearize about hover, giving three decoupled subsystems:
- Position [x,y,ẋ,ẏ] → [φ_cmd, θ_cmd]
- Attitude [φ,θ,ψ,p,q,r] → [U₂,U₃,U₄]
- Altitude [z,ż] → δU₁
"""
function linearize_hover(p::QuadrotorParams)
    g, m = p.g, p.m
    Ix, Iy, Iz = p.Ix, p.Iy, p.Iz
    l, Jr = p.l, p.Jr
    kf = p.kf

    ω0 = sqrt(m * g / (4 * kf))
    Ωr0 = 0.0

    A_pos = [0.0 0.0 1.0 0.0;
             0.0 0.0 0.0 1.0;
             0.0 0.0 0.0 0.0;
             0.0 0.0 0.0 0.0]

    B_pos = [0.0  0.0;
             0.0  0.0;
             0.0  g;
             -g   0.0]

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

    B_z = reshape([0.0; 1.0/m], 2, 1)

    return A_pos, B_pos, A_att, B_att, A_z, B_z
end

"""
    assemble_full_plant(p) -> (A, B, U_eq)

Embed decoupled subsystems into the full 12x12 state and 12x4 input matrices.
State ordering: x = [x,y,z, ẋ,ẏ,ż, φ,θ,ψ, p,q,r], inputs U = [U₁,U₂,U₃,U₄].
Linearized error dynamics: ẋ = A·x + B·(U - U_eq), U_eq = [mg,0,0,0].
"""
function assemble_full_plant(p::QuadrotorParams = DEFAULT_PARAMS)
    A_pos, B_pos, A_att, B_att, A_z, B_z = linearize_hover(p)

    A = zeros(12, 12)
    A[1, 4] = 1.0
    A[2, 5] = 1.0
    A[3, 6] = 1.0
    A[4, 8] = p.g
    A[5, 7] = -p.g
    A[7, 10] = 1.0
    A[8, 11] = 1.0
    A[9, 12] = 1.0
    A[10, 11] = A_att[4, 5]
    A[11, 10] = A_att[5, 4]

    B = zeros(12, 4)
    B[6, 1] = 1.0 / p.m
    B[10, 2] = p.l / p.Ix
    B[11, 3] = p.l / p.Iy
    B[12, 4] = 1.0 / p.Iz

    U_eq = [p.m * p.g, 0.0, 0.0, 0.0]

    return A, B, U_eq
end

"""
    LQRController

Pre-computed cascade LQR gains. K_pos (2x4), K_att (3x6), K_z (1x2).
"""
struct LQRController
    K_pos::Matrix{Float64}
    K_att::Matrix{Float64}
    K_z::Matrix{Float64}
    params::QuadrotorParams
end

# Default weights from Khan et al. 2024
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

    for (name, A, B) in (("position", A_pos, B_pos),
                          ("attitude", A_att, B_att),
                          ("altitude", A_z, B_z))
        r = rank(ctrb(A, B))
        r == size(A, 1) || @warn "$name subsystem is NOT controllable (rank = $r)"
    end

    K_pos = lqr(A_pos, B_pos, Matrix(Q_pos), Matrix(R_pos))
    K_att = lqr(A_att, B_att, Matrix(Q_att), Matrix(R_att))
    K_z = lqr(A_z, B_z, Matrix(Q_z), Matrix(R_z))

    return LQRController(K_pos, K_att, K_z, p)
end

"""
    compute_control(ctrl, x, x_ref) -> (ω², U)

Cascade control law u = -K(x - x_ref), then mixer inversion M·ω² = U.
Position loop sets attitude corrections; attitude loop sets torques.
"""
function compute_control(ctrl::LQRController,
                         x::AbstractVector, x_ref::AbstractVector)
    p = ctrl.params

    e_pos = x[[1,2,4,5]] - x_ref[[1,2,4,5]]
    e_z = x[[3,6]] - x_ref[[3,6]]

    φθ_cmd = -ctrl.K_pos * e_pos
    δU1 = (-ctrl.K_z * e_z)[1]

    att_ref = copy(x_ref[7:12])
    att_ref[1:2] .+= φθ_cmd
    U_att = -ctrl.K_att * (x[7:12] - att_ref)

    U1 = p.m * p.g + δU1
    U2, U3, U4 = U_att

    kf, km, l = p.kf, p.km, p.l
    M = [kf    kf      kf     kf;
         0.0  -kf*l    0.0    kf*l;
         kf*l  0.0    -kf*l   0.0;
        -km    km     -km     km]

    ω² = M \ [U1; U2; U3; U4]

    return ω², [U1, U2, U3, U4]
end

"""
    PIDController

Diagonal cascaded PD/PID controller. Gains are extracted from the LQR solution
for each decoupled subsystem (dropping MIMO cross-coupling) so the PID has the
best possible single-channel tuning while still being structurally simpler than LQR.
"""
mutable struct PIDController
    Kp_pos::Vector{Float64} # [x-channel, y-channel]
    Kd_pos::Vector{Float64}
    Kp_z::Float64
    Ki_z::Float64
    Kd_z::Float64
    Kp_att::Vector{Float64} # [φ, θ, ψ]
    Kd_att::Vector{Float64}
    e_int_z::Float64        # altitude integrator state
    dt::Float64
    params::QuadrotorParams
end

function PIDController(
    p::QuadrotorParams = DEFAULT_PARAMS;
    Q_pos::AbstractMatrix = Diagonal([0.1, 0.1, 0.001, 0.001]),
    R_pos::AbstractMatrix = Diagonal([1.0, 1.0]),
    Q_att::AbstractMatrix = Diagonal([20.0, 17.0, 0.15, 0.05, 0.05, 0.09]),
    R_att::AbstractMatrix = Diagonal([1.0, 1.0, 1.0]),
    Q_z::AbstractMatrix = Diagonal([0.03, 0.05]),
    R_z::AbstractMatrix = Diagonal([0.0002]),
    Ki_z::Float64 = 0.0,
    dt::Float64 = 1e-3,
)
    A_pos, B_pos, A_att, B_att, A_z, B_z = linearize_hover(p)
    K_pos = lqr(A_pos, B_pos, Matrix(Q_pos), Matrix(R_pos))
    K_att = lqr(A_att, B_att, Matrix(Q_att), Matrix(R_att))
    K_z = lqr(A_z, B_z, Matrix(Q_z), Matrix(R_z))

    # B_pos[:,1] = φ_cmd → ẏ;  B_pos[:,2] = θ_cmd → ẋ
    # K_pos row 1 → φ_cmd (controls y);  row 2 → θ_cmd (controls x)
    Kp_pos = [K_pos[2, 1], K_pos[1, 2]]
    Kd_pos = [K_pos[2, 3], K_pos[1, 4]]

    # K_att rows: [U₂, U₃, U₄];  cols: [φ, θ, ψ, p, q, r]
    Kp_att = [K_att[1, 1], K_att[2, 2], K_att[3, 3]]
    Kd_att = [K_att[1, 4], K_att[2, 5], K_att[3, 6]]

    return PIDController(Kp_pos, Kd_pos, K_z[1, 1], Ki_z, K_z[1, 2], Kp_att, Kd_att, 0.0, dt, p)
end

"""
    compute_control(ctrl::PIDController, x, x_ref) -> (ω², U)

Diagonal cascaded PD control. Uses state velocities/rates directly (no finite
differences). Position loop outputs angle commands fed into the attitude loop.
"""
function compute_control(ctrl::PIDController,
                         x::AbstractVector, x_ref::AbstractVector)
    p = ctrl.params

    # Position PD (ẋ, ẏ are states 4, 5)
    e_x = x[1] - x_ref[1];  ė_x = x[4] - x_ref[4]
    e_y = x[2] - x_ref[2];  ė_y = x[5] - x_ref[5]
    θ_cmd = -(ctrl.Kp_pos[1] * e_x + ctrl.Kd_pos[1] * ė_x)
    φ_cmd = -(ctrl.Kp_pos[2] * e_y + ctrl.Kd_pos[2] * ė_y)

    # Altitude PID (ż is state 6)
    e_z = x[3] - x_ref[3];  ė_z = x[6] - x_ref[6]
    ctrl.e_int_z += e_z * ctrl.dt
    δU1 = -(ctrl.Kp_z * e_z + ctrl.Ki_z * ctrl.e_int_z + ctrl.Kd_z * ė_z)

    # Attitude PD (rates p, q, r are states 10, 11, 12)
    att_ref = copy(x_ref[7:12])
    att_ref[1:2] .+= [φ_cmd, θ_cmd]
    e_ang  = x[7:9] - att_ref[1:3]
    e_rate = x[10:12] - att_ref[4:6]
    U2 = -(ctrl.Kp_att[1] * e_ang[1] + ctrl.Kd_att[1] * e_rate[1])
    U3 = -(ctrl.Kp_att[2] * e_ang[2] + ctrl.Kd_att[2] * e_rate[2])
    U4 = -(ctrl.Kp_att[3] * e_ang[3] + ctrl.Kd_att[3] * e_rate[3])

    U1 = p.m * p.g + δU1

    kf, km, l = p.kf, p.km, p.l
    M = [kf    kf      kf     kf;
         0.0  -kf*l    0.0    kf*l;
         kf*l  0.0    -kf*l   0.0;
        -km    km     -km     km]
    ω² = M \ [U1; U2; U3; U4]

    return ω², [U1, U2, U3, U4]
end

"""
    SheafControllerInterface

Mediates time-scale separation between the slow sheaf planner (≤10 Hz) and
the fast inner loop (~1 kHz). Controller-agnostic: works with LQRController,
PIDController, or any type that implements compute_control(ctrl, x, x_ref).
"""
mutable struct SheafControllerInterface
    ctrl  # LQRController, PIDController, or any compatible type
    x_ref::Vector{Float64}
    t_last_plan::Float64
    planner_dt::Float64
end

function SheafControllerInterface(ctrl; planner_hz::Float64 = 10.0)
    return SheafControllerInterface(ctrl, zeros(12), -Inf, 1.0 / planner_hz)
end

function update_reference!(iface::SheafControllerInterface,
                            x_ref_new::AbstractVector, t_now::Float64)
    @assert length(x_ref_new) == 12 "x_ref must be a 12-element state vector"
    copyto!(iface.x_ref, x_ref_new)
    iface.t_last_plan = t_now
    return iface
end

"""
    step!(iface, x) -> (ω², U)

Inner-loop tick (~1 kHz): evaluate the inner-loop control law.
"""
function step!(iface::SheafControllerInterface, x::AbstractVector)
    return compute_control(iface.ctrl, x, iface.x_ref)
end

"""
    formation_coboundary(n_agents, edges, offsets) -> (D, b)

Build coboundary matrix D ∈ ℝ^{3mx3n} and offset vector b for a formation
sheaf over a graph with n agents and m edges. Formation is satisfied when Dx = b.
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
Slow outer planner (≤10 Hz) sets position references via `sheaf_plan_step!`;
fast inner loop (~1 kHz) tracks them via `swarm_step!`.
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
subspace {x : Dx = b} via conjugate gradients and updates each agent's reference.
"""
function sheaf_plan_step!(
    coord::SwarmCoordinator,
    states::Vector{<:AbstractVector},
    t_now::Float64,
)
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
    swarm_step!(coord, states) -> Vector{Tuple}

Inner-loop tick (~1 kHz): evaluate u = -K(x - x_ref) for each agent.
"""
function swarm_step!(coord::SwarmCoordinator, states::Vector{<:AbstractVector})
    @assert length(states) == coord.n_agents "expected $(coord.n_agents) state vectors"
    return [step!(coord.controllers[i], states[i]) for i in 1:coord.n_agents]
end

"""
    SimRecord

Per-agent simulation history. Fields: t, x (12xn), x_ref (12xn), U (4xn), omega2 (4xn).
"""
struct SimRecord
    t::Vector{Float64}
    x::Matrix{Float64}
    x_ref::Matrix{Float64}
    U::Matrix{Float64}
    omega2::Matrix{Float64}
end

# RK4 integration of linearized plant ẋ = A·x + B·(u - U_eq).
function _rk4_step(A::Matrix, B::Matrix, U_eq::Vector,
                   x::AbstractVector, u::AbstractVector, dt::Float64)
    δu = u - U_eq
    f(s) = A * s + B * δu
    k1 = f(x)
    k2 = f(x + dt/2 * k1)
    k3 = f(x + dt/2 * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6) .* (k1 + 2k2 + 2k3 + k4)
end

"""
    run_baseline_sim(waypoints; params, x0s, dt, t_end, planner_hz) -> Vector{SimRecord}

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

    A_full, B_full, U_eq = assemble_full_plant(params)
    agent_ctrls = [deepcopy(ctrl) for _ in 1:n_agents]

    records = [SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 4, n_steps),
        Matrix{Float64}(undef, 4, n_steps),
    ) for _ in 1:n_agents]

    states = [copy(x0s[i]) for i in 1:n_agents]

    for k in 1:n_steps
        t_now = (k - 1) * dt
        for i in 1:n_agents
            x = states[i]
            x_ref = waypoints[i]
            ω², U = compute_control(agent_ctrls[i], x, x_ref)
            records[i].t[k] = t_now
            records[i].x[:, k] = x
            records[i].x_ref[:, k] = x_ref
            records[i].U[:, k] = U
            records[i].omega2[:, k] = ω²
            states[i] = _rk4_step(A_full, B_full, U_eq, x, U, dt)
        end
    end

    return records
end

"""
    run_coordinated_sim(n_agents, edges, offsets; ...) -> Vector{SimRecord}

Sheaf-coordinated simulation. Outer planner updates position references at
`planner_hz`; each agent's LQR inner loop runs at 1/dt.
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

    A_full, B_full, U_eq = assemble_full_plant(params)

    coord = SwarmCoordinator(n_agents, edges, offsets;
                             params=params, ctrl=ctrl, planner_hz=planner_hz)

    for i in 1:n_agents
        update_reference!(coord.controllers[i], copy(x0s[i]), 0.0)
    end

    records = [SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 12, n_steps),
        Matrix{Float64}(undef, 4, n_steps),
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
            ω², U = results[i]
            records[i].t[k] = t_now
            records[i].x[:, k] = states[i]
            records[i].x_ref[:, k] = coord.controllers[i].x_ref
            records[i].U[:, k] = U
            records[i].omega2[:, k] = ω²
            states[i] = _rk4_step(A_full, B_full, U_eq, states[i], U, dt)
        end
    end

    return records
end

"""
    plot_trajectories(records; title) -> Plot

x, y, z time series for all agents. Dashed lines show references.
"""
function plot_trajectories(records::Vector{SimRecord};
                           title::String = "Position trajectories")
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
    n_steps = length(t)
    err = [norm(D * vcat([records[i].x[1:3, k] for i in eachindex(records)]...) - b)
           for k in 1:n_steps]

    return plot(t, err;
                xlabel="t [s]",
                ylabel="‖Dx - b‖  [m]",
                title=title,
                legend=false,
                linewidth=2,
                color=:steelblue,
                size=(800, 300),
                left_margin=10Plots.mm, bottom_margin=8Plots.mm)
end

"""
    plot_motor_commands(record, agent_id; title) -> Plot

Virtual inputs U = [U₁,U₂,U₃,U₄] and squared motor speeds ω² for one agent.
"""
function plot_motor_commands(record::SimRecord, agent_id::Int = 1;
                             title::String = "Motor commands — Agent $agent_id")
    t = record.t
    p1 = plot(t, record.U';
              labels=["U₁" "U₂" "U₃" "U₄"],
              ylabel="Virtual input [N or N·m]",
              xlabel="",
              title=title,
              linewidth=1.5,
              legend=:topright,
              left_margin=10Plots.mm)
    p2 = plot(t, record.omega2';
              labels=["ω₁²" "ω₂²" "ω₃²" "ω₄²"],
              ylabel="ω² [rad²/s²]",
              xlabel="t [s]",
              linewidth=1.5,
              legend=:topright,
              left_margin=10Plots.mm, bottom_margin=8Plots.mm)
    return plot(p1, p2; layout=(2, 1), size=(800, 500), link=:x)
end

"""
    compare_runs(run1, run2, D, b; title, label1, label2) -> Plot

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
    pz_base  = plot(title="$label1 — z [m]", xlabel="", ylabel="z [m]", legend=:bottomright, left_margin=margin)
    pz_coord = plot(title="$label2 — z [m]", xlabel="", ylabel="z [m]", legend=:bottomright, left_margin=margin)
    pe_base  = plot(title="$label1 — formation error", xlabel="t [s]", ylabel="‖Dx-b‖ [m]", legend=false, color=:crimson, linewidth=2, left_margin=margin, bottom_margin=margin)
    pe_coord = plot(title="$label2 — formation error", xlabel="t [s]", ylabel="‖Dx-b‖ [m]", legend=false, color=:steelblue, linewidth=2, left_margin=margin, bottom_margin=margin)

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
