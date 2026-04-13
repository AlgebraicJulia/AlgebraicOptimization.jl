using AlgebraicOptimization
using Plots
using LinearAlgebra
using Dates

# Agents 1-2 are submarines; agents 3-4 are quadrotors.
# Each quadrotor hovers 3 m above its paired submarine, forming a rectangle:
#
#   quad3 ─────── quad4       z ≈ +1.5 m
#     |               |
#   sub1  ─────── sub2        z ≈ −1.5 m
#
# Starting from all zeros, the sheaf nearest-section pushes the submarines
# slightly below the surface and the quadrotors into the air.

# Coboundary sign convention: offset = x[v1] - x[v2].
# To place quad (v2) 3 m above sub (v1): z_sub - z_quad = -3.
edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
offsets = Dict(
    (1, 2) => [ 2.0, 0.0,  0.0],   # sub1 - sub2  in x: sub1 is 2 m ahead
    (1, 3) => [ 0.0, 0.0, -3.0],   # sub1 - quad3 in z: quad3 is 3 m above
    (2, 4) => [ 0.0, 0.0, -3.0],   # sub2 - quad4 in z: quad4 is 3 m above
    (3, 4) => [ 2.0, 0.0,  0.0],   # quad3 - quad4 in x: quad3 is 2 m ahead
)

n_agents = 4
sub_ids = [1, 2]
quad_ids = [3, 4]

D, b = formation_coboundary(n_agents, edges, offsets)

sub_model = SubmarineModel()
quad_model = QuadrotorModel()

models = [sub_model, sub_model, quad_model, quad_model]
ctrls = [LinearLQRController(sub_model),
        LinearLQRController(sub_model),
        LQRController(),
        LQRController()]

x0s = [zeros(state_dim(m)) for m in models]

println("Running heterogeneous coordinated simulation (submarines + quadrotors)...")
runs = run_coordinated_sim(models, ctrls, edges, offsets; x0s=x0s, t_end=60.0)

err = norm(D * vcat([runs[i].x[1:3, end] for i in 1:n_agents]...) - b)
println("\nFormation error at t = 60 s: ", round(err, sigdigits=4), " m")

"""
    plot_heterogeneous_trajectories(runs, sub_ids, quad_ids)

2x2 grid: rows = x, z (y omitted — formation is in the x-z plane, y ≈ 0);
left column = submarines, right column = quadrotors.
layout=(2,2) is row-major: p[1]=top-left, p[2]=top-right, p[3]=bot-left, p[4]=bot-right.
Solid lines are actual trajectories; dashed lines are sheaf references.
"""
function plot_heterogeneous_trajectories(
    runs::Vector,
    sub_ids::Vector{Int},
    quad_ids::Vector{Int},
)
    t = runs[1].t
    sub_cols = palette(:Blues_9,   length(sub_ids)  + 4)[5:end]
    quad_cols = palette(:Oranges_9, length(quad_ids) + 4)[5:end]

    # row-major (2,2): p[1]=top-left, p[2]=top-right, p[3]=bot-left, p[4]=bot-right
    p = plot(layout=(2, 2), size=(900, 550), link=:x,
             plot_title="Heterogeneous formation trajectories",
             left_margin=12Plots.mm, bottom_margin=8Plots.mm,
             top_margin=4Plots.mm, right_margin=4Plots.mm)

    for (row, (lbl, idx)) in enumerate([("x [m]", 1), ("z [m]", 3)])
        sp_sub = (row - 1) * 2 + 1   # col 1: p[1], p[3]
        sp_quad = (row - 1) * 2 + 2   # col 2: p[2], p[4]

        plot!(p[sp_sub]; ylabel=lbl, xlabel=row==2 ? "t [s]" : "",
              title=row==1 ? "Submarines"  : "", legend=row==1 ? :right : false,
              guidefontsize=10, tickfontsize=8)
        plot!(p[sp_quad]; ylabel=lbl, xlabel=row==2 ? "t [s]" : "",
              title=row==1 ? "Quadrotors" : "", legend=row==1 ? :right : false,
              guidefontsize=10, tickfontsize=8)

        for (j, id) in enumerate(sub_ids)
            plot!(p[sp_sub], t, runs[id].x[idx, :];
                label=row==1 ? "Sub $id" : "", color=sub_cols[j], linewidth=1.5)
            plot!(p[sp_sub], t, runs[id].x_ref[idx, :];
                label="", color=sub_cols[j], linestyle=:dash, alpha=0.5)
        end
        for (j, id) in enumerate(quad_ids)
            plot!(p[sp_quad], t, runs[id].x[idx, :];
                label=row==1 ? "Quad $id" : "", color=quad_cols[j], linewidth=1.5)
            plot!(p[sp_quad], t, runs[id].x_ref[idx, :];
                label="", color=quad_cols[j], linestyle=:dash, alpha=0.5)
        end
    end
    return p
end

"""
    plot_formation_snapshot(runs, edges, sub_ids, quad_ids)

x-z side view of agent positions at t = 0 (hollow) and t = end (filled).
Formation edges are drawn at the final time. A dashed line marks z = 0
(air/sea interface).
"""
function plot_formation_snapshot(
    runs::Vector,
    edges::Vector{Tuple{Int,Int}},
    sub_ids::Vector{Int},
    quad_ids::Vector{Int},
)
    pos_start = [runs[i].x[1:3, 1] for i in eachindex(runs)]
    pos_end = [runs[i].x[1:3, end] for i in eachindex(runs)]

    p = plot(size=(600, 500), xlabel="x [m]", ylabel="z [m]",
             title="Formation snapshot (x–z plane)",
             legend=:outertopright, aspect_ratio=:equal,
             left_margin=10Plots.mm, bottom_margin=8Plots.mm)

    # Air/sea interface
    xlims_pad = 1.5
    x_all = vcat([pos_end[i][1] for i in eachindex(runs)],
                 [pos_start[i][1] for i in eachindex(runs)])
    hline!(p, [0.0]; linestyle=:dash, color=:steelblue, alpha=0.5,
           label="surface (z = 0)", linewidth=1)

    # Formation edges at final time
    for (v1, v2) in edges
        plot!(p, [pos_end[v1][1], pos_end[v2][1]],
                 [pos_end[v1][3], pos_end[v2][3]];
              color=:gray, linewidth=1, linestyle=:dot, label="")
    end

    # Agent positions: hollow = start, filled = end
    sub_marker = :dtriangle   # pointing down → underwater
    quad_marker = :utriangle   # pointing up   → airborne

    for (k, id) in enumerate(sub_ids)
        scatter!(p, [pos_start[id][1]], [pos_start[id][3]];
            marker=sub_marker, markersize=8, color=:steelblue,
            markerstrokewidth=2, markerstrokecolor=:steelblue,
            markeralpha=0.0, label=k==1 ? "Sub (t=0)" : "")
        scatter!(p, [pos_end[id][1]], [pos_end[id][3]];
            marker=sub_marker, markersize=10, color=:steelblue,
            label=k==1 ? "Sub (t=end)" : "")
    end

    for (k, id) in enumerate(quad_ids)
        scatter!(p, [pos_start[id][1]], [pos_start[id][3]];
            marker=quad_marker, markersize=8, color=:darkorange,
            markerstrokewidth=2, markerstrokecolor=:darkorange,
            markeralpha=0.0, label=k==1 ? "Quad (t=0)" : "")
        scatter!(p, [pos_end[id][1]], [pos_end[id][3]];
            marker=quad_marker, markersize=10, color=:darkorange,
            label=k==1 ? "Quad (t=end)" : "")
    end

    return p
end

"""
    plot_xz_paths(runs, edges, sub_ids, quad_ids)

x-z plane view showing the full trajectory path of each agent from start to
end, with start (hollow) and end (filled) markers, formation edges at the
final time, and a dashed surface line at z = 0.
"""
function plot_xz_paths(
    runs::Vector,
    edges::Vector{Tuple{Int,Int}},
    sub_ids::Vector{Int},
    quad_ids::Vector{Int},
)
    sub_cols  = palette(:Blues_9,   length(sub_ids)  + 4)[5:end]
    quad_cols = palette(:Oranges_9, length(quad_ids) + 4)[5:end]

    p = plot(size=(650, 500), xlabel="x [m]", ylabel="z [m]",
             title="Formation paths (x-z plane)",
             legend=:outertopright, aspect_ratio=:equal,
             left_margin=10Plots.mm, bottom_margin=8Plots.mm)

    # Surface line
    hline!(p, [0.0]; linestyle=:dash, color=:royalblue, alpha=0.4,
           label="surface (z = 0)", linewidth=1)

    # Trajectory paths
    for (j, id) in enumerate(sub_ids)
        plot!(p, runs[id].x[1, :], runs[id].x[3, :];
            color=sub_cols[j], linewidth=1.5, label="Sub $id")
    end
    for (j, id) in enumerate(quad_ids)
        plot!(p, runs[id].x[1, :], runs[id].x[3, :];
            color=quad_cols[j], linewidth=1.5, label="Quad $id")
    end

    # Formation edges at final time
    pos_end = [runs[i].x[1:3, end] for i in eachindex(runs)]
    for (v1, v2) in edges
        plot!(p, [pos_end[v1][1], pos_end[v2][1]],
                 [pos_end[v1][3], pos_end[v2][3]];
              color=:gray, linewidth=1, linestyle=:dot, label="")
    end

    # Start markers (hollow) and end markers (filled)
    pos_start = [runs[i].x[1:3, 1] for i in eachindex(runs)]
    for (j, id) in enumerate(sub_ids)
        scatter!(p, [pos_start[id][1]], [pos_start[id][3]];
            marker=:dtriangle, markersize=8, color=sub_cols[j],
            markeralpha=0.0, markerstrokewidth=2,
            markerstrokecolor=sub_cols[j], label="")
        scatter!(p, [pos_end[id][1]], [pos_end[id][3]];
            marker=:dtriangle, markersize=10, color=sub_cols[j], label="")
    end
    for (j, id) in enumerate(quad_ids)
        scatter!(p, [pos_start[id][1]], [pos_start[id][3]];
            marker=:utriangle, markersize=8, color=quad_cols[j],
            markeralpha=0.0, markerstrokewidth=2,
            markerstrokecolor=quad_cols[j], label="")
        scatter!(p, [pos_end[id][1]], [pos_end[id][3]];
            marker=:utriangle, markersize=10, color=quad_cols[j], label="")
    end

    return p
end

# ── Save plots ────────────────────────────────────────────────────────────────

println("\nSaving plots...")
fp = "examples/heterogenous-lqr/figures/"
isdir(fp) || mkdir(fp)
date = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

savefig(
    plot_heterogeneous_trajectories(runs, sub_ids, quad_ids),
    fp * "trajectories_" * date * ".png",
)

savefig(
    plot_formation_snapshot(runs, edges, sub_ids, quad_ids),
    fp * "snapshot_" * date * ".png",
)

savefig(
    plot_xz_paths(runs, edges, sub_ids, quad_ids),
    fp * "paths_xz_" * date * ".png",
)

savefig(
    plot_formation_error(runs, D, b; title="Heterogeneous formation error"),
    fp * "formation_error_" * date * ".png",
)

println("Done. Plots written to " * fp)
