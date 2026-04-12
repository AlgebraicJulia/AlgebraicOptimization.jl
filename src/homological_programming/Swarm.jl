module Swarm

using LinearAlgebra
using ..CellularSheaves
using ..VehicleInterface
using ..Controllers

export formation_coboundary
export SwarmCoordinator, sheaf_plan_step!, swarm_step!
export run_baseline_sim, run_coordinated_sim

"""
    formation_coboundary(n_agents, edges, offsets; pos_dim) -> (D, b)

Build coboundary matrix D ∈ ℝ^{pos_dim·m × pos_dim·n} and offset vector b for
a position formation sheaf over a graph with n agents and m edges. Formation is
satisfied when Dx = b. `pos_dim` is the dimension of the shared position space
(default 3).

Ref: Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves",
Journal of Applied and Computational Topology, 2019.
"""
function formation_coboundary(
    n_agents::Int,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}} = Dict{Tuple{Int,Int}, Vector{Float64}}();
    pos_dim::Int = 3,
)
    sheaf, b = _formation_sheaf(n_agents, edges, offsets, pos_dim)
    D = Matrix{Float64}(coboundary_map(sheaf))
    return D, b
end

function _formation_sheaf(
    n_agents::Int,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}},
    pos_dim::Int,
)
    n_edges = length(edges)
    sheaf = CellularSheaf(fill(pos_dim, n_agents), fill(pos_dim, n_edges))
    b = zeros(pos_dim * n_edges)
    Ip = Matrix(1.0 * I, pos_dim, pos_dim)
    for (e, (v1, v2)) in enumerate(edges)
        set_edge_maps!(sheaf, v1, v2, e, Ip, Ip)
        b[pos_dim*(e-1)+1:pos_dim*e] = get(offsets, (v1, v2), zeros(pos_dim))
    end
    return sheaf, b
end

"""
    SwarmCoordinator

N-agent swarm using a cellular sheaf Laplacian for formation coordination.
Vehicle-agnostic: each agent can be a different model type with its own
controller. All agents must share the same position space dimension.

Slow outer planner sets position references via `sheaf_plan_step!`;
fast inner loop tracks them via `swarm_step!`.
"""
mutable struct SwarmCoordinator
    sheaf::CellularSheaf
    b::Vector{Float64}
    controllers::Vector{SheafControllerInterface}
    pos_indices::Vector{UnitRange{Int}}
    pos_dim::Int
    n_agents::Int
end

"""
    SwarmCoordinator(models, ctrls, edges, offsets; planner_hz) -> SwarmCoordinator

Construct a coordinator for a heterogeneous swarm. `models` and `ctrls` are
per-agent vectors. All agents must expose the same position dimension via
`position_indices`.
"""
function SwarmCoordinator(
    models::Vector{<:AbstractVehicleModel},
    ctrls::Vector,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}} = Dict{Tuple{Int,Int}, Vector{Float64}}();
    planner_hz::Float64 = 10.0,
)
    n_agents = length(models)
    @assert length(ctrls) == n_agents "expected $n_agents controllers, got $(length(ctrls))"

    pos_idx = [position_indices(m) for m in models]
    pos_dims = [length(idx) for idx in pos_idx]
    @assert allequal(pos_dims) "all agents must share the same position space dimension"
    pos_dim = pos_dims[1]

    sheaf, b = _formation_sheaf(n_agents, edges, offsets, pos_dim)
    controllers = [SheafControllerInterface(deepcopy(ctrls[i]);
                                            n_states=state_dim(models[i]),
                                            planner_hz=planner_hz)
                   for i in 1:n_agents]
    return SwarmCoordinator(sheaf, b, controllers, pos_idx, pos_dim, n_agents)
end

"""
    sheaf_plan_step!(coord, states, t_now) -> pos_new

Outer planner tick. Projects current positions onto the formation-consistent
subspace {x : Dx = b} and updates each agent's reference. Position is
extracted using each agent's `position_indices`.
"""
function sheaf_plan_step!(
    coord::SwarmCoordinator,
    states::Vector{<:AbstractVector},
    t_now::Float64,
)
    @assert length(states) == coord.n_agents "expected $(coord.n_agents) state vectors"

    pos = vcat([states[i][coord.pos_indices[i]] for i in 1:coord.n_agents]...)
    pos_new = Vector{Float64}(nearest_section(coord.sheaf, pos, coord.b))

    for i in 1:coord.n_agents
        x_ref_new = copy(coord.controllers[i].x_ref)
        x_ref_new[coord.pos_indices[i]] .= pos_new[coord.pos_dim*(i-1)+1:coord.pos_dim*i]
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
    run_baseline_sim(models, ctrls, waypoints; x0s, dt, t_end) -> Vector{SimRecord}

Each agent independently tracks its own fixed waypoint with no inter-agent
coupling. Models and controllers are per-agent, so vehicle types may differ.
"""
function run_baseline_sim(
    models::Vector{<:AbstractVehicleModel},
    ctrls::Vector,
    waypoints::Vector{<:AbstractVector};
    x0s::Vector{<:AbstractVector} = [zeros(state_dim(m)) for m in models],
    dt::Float64 = 1e-3,
    t_end::Float64 = 10.0,
)
    n_agents = length(models)
    @assert length(ctrls) == n_agents "expected $n_agents controllers"
    @assert length(waypoints) == n_agents "expected $n_agents waypoints"

    n_steps = round(Int, t_end / dt)
    lin_models = [LinearizedModel(m) for m in models]

    records = [SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, state_dim(models[i]), n_steps),
        Matrix{Float64}(undef, state_dim(models[i]), n_steps),
        Matrix{Float64}(undef, control_dim(models[i]), n_steps),
    ) for i in 1:n_agents]

    states = [copy(Vector{Float64}(x0s[i])) for i in 1:n_agents]

    for k in 1:n_steps
        t_now = (k - 1) * dt
        for i in 1:n_agents
            x = states[i]
            x_ref = waypoints[i]
            u = compute_control(ctrls[i], x, x_ref)
            records[i].t[k] = t_now
            records[i].x[:, k] = x
            records[i].x_ref[:, k] = x_ref
            records[i].u[:, k] = u
            states[i] = _rk4(lin_models[i], x, u, dt)
        end
    end

    return records
end

"""
    run_coordinated_sim(models, ctrls, edges, offsets; x0s, dt, t_end, planner_hz) -> Vector{SimRecord}

Sheaf-coordinated simulation. Outer planner updates position references at
`planner_hz`; each agent's inner loop runs at 1/dt. Models and controllers
are per-agent, so vehicle types may differ.
"""
function run_coordinated_sim(
    models::Vector{<:AbstractVehicleModel},
    ctrls::Vector,
    edges::Vector{Tuple{Int,Int}},
    offsets::Dict{Tuple{Int,Int}, Vector{Float64}} = Dict{Tuple{Int,Int}, Vector{Float64}}();
    x0s::Vector{<:AbstractVector} = [zeros(state_dim(m)) for m in models],
    dt::Float64 = 1e-3,
    t_end::Float64 = 10.0,
    planner_hz::Float64 = 10.0,
)
    n_agents = length(models)
    n_steps = round(Int, t_end / dt)
    planner_ticks = round(Int, (1.0 / planner_hz) / dt)
    lin_models = [LinearizedModel(m) for m in models]

    coord = SwarmCoordinator(models, ctrls, edges, offsets; planner_hz=planner_hz)

    for i in 1:n_agents
        update_reference!(coord.controllers[i], copy(x0s[i]), 0.0)
    end

    records = [SimRecord(
        Vector{Float64}(undef, n_steps),
        Matrix{Float64}(undef, state_dim(models[i]), n_steps),
        Matrix{Float64}(undef, state_dim(models[i]), n_steps),
        Matrix{Float64}(undef, control_dim(models[i]), n_steps),
    ) for i in 1:n_agents]

    states = [copy(Vector{Float64}(x0s[i])) for i in 1:n_agents]

    for k in 1:n_steps
        t_now = (k - 1) * dt

        if mod(k - 1, planner_ticks) == 0
            sheaf_plan_step!(coord, states, t_now)
        end

        results = swarm_step!(coord, states)

        for i in 1:n_agents
            u = results[i]
            records[i].t[k] = t_now
            records[i].x[:, k] = states[i]
            records[i].x_ref[:, k] = coord.controllers[i].x_ref
            records[i].u[:, k] = u
            states[i] = _rk4(lin_models[i], states[i], u, dt)
        end
    end

    return records
end

end
