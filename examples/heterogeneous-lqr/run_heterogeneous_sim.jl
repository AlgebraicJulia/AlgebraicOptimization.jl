using AlgebraicOptimization
using Plots
using LinearAlgebra
using Dates

# Palette matching PaperPlotting.jl
const _pp_blue   = RGB(97/255,  136/255, 178/255)
const _pp_orange = RGB(223/255, 167/255, 119/255)
const _pp_green  = RGB(172/255, 207/255, 146/255)
const _pp_purple = RGB(216/255, 201/255, 238/255)
const _pp_rose   = RGB(215/255, 130/255, 140/255)

# Per-type palettes — extend if more agents are ever added
const _sub_palette  = [_pp_blue,   _pp_purple]
const _quad_palette = [_pp_orange, _pp_green, _pp_rose]

# ── H1: 2 submarines + 2 quadrotors, rectangular formation ───────────────────
#
#   quad3 ─────── quad4       z ≈ +1.5 m
#     |               |
#   sub1  ─────── sub2        z ≈ −1.5 m

edges_h1 = [(1, 2), (1, 3), (2, 4), (3, 4)]
offsets_h1 = Dict(
    (1, 2) => [ 2.0, 0.0,  0.0],   # sub1 - sub2:   sub1 is 2 m ahead
    (1, 3) => [ 0.0, 0.0, -3.0],   # sub1 - quad3:  quad3 is 3 m above
    (2, 4) => [ 0.0, 0.0, -3.0],   # sub2 - quad4:  quad4 is 3 m above
    (3, 4) => [ 2.0, 0.0,  0.0],   # quad3 - quad4: quad3 is 2 m ahead
)

sub_ids_h1  = [1, 2]
quad_ids_h1 = [3, 4]
n_h1 = 4

D_h1, b_h1 = formation_coboundary(n_h1, edges_h1, offsets_h1)

sub_model  = SubmarineModel()
quad_model = QuadrotorModel()

models_h1 = [sub_model, sub_model, quad_model, quad_model]
ctrls_h1  = [LinearLQRController(sub_model),
             LinearLQRController(sub_model),
             LQRController(),
             LQRController()]
x0s_h1 = [zeros(state_dim(m)) for m in models_h1]

println("Running H1: 2 subs + 2 quads (rectangular formation)...")
runs_h1 = run_coordinated_sim(models_h1, ctrls_h1, edges_h1, offsets_h1;
                               x0s=x0s_h1, t_end=60.0)
err_h1 = norm(D_h1 * vcat([runs_h1[i].x[1:3, end] for i in 1:n_h1]...) - b_h1)
println("  Formation error at t = 60 s: ", round(err_h1, sigdigits=4), " m")

# ── H2: 2 submarines + 3 quadrotors, pyramid formation ───────────────────────
#
#           quad5            z ≈ +4.5 m  (apex)
#          /     \
#     quad3 ─── quad4        z ≈ +3.0 m
#       |           |
#     sub1 ─────── sub2      z ≈  0.0 m

edges_h2 = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
offsets_h2 = Dict(
    (1, 2) => [-3.0, 0.0,  0.0],   # sub1 - sub2:   sub2 is 3 m to the right
    (1, 3) => [ 0.0, 0.0, -3.0],   # sub1 - quad3:  quad3 is 3 m above sub1
    (2, 4) => [ 0.0, 0.0, -3.0],   # sub2 - quad4:  quad4 is 3 m above sub2
    (3, 5) => [-1.5, 0.0, -1.5],   # quad3 - quad5: apex 1.5 m right and 1.5 m above
    (4, 5) => [ 1.5, 0.0, -1.5],   # quad4 - quad5: apex 1.5 m left and 1.5 m above
)

sub_ids_h2  = [1, 2]
quad_ids_h2 = [3, 4, 5]
n_h2 = 5

D_h2, b_h2 = formation_coboundary(n_h2, edges_h2, offsets_h2)

models_h2 = [sub_model, sub_model, quad_model, quad_model, quad_model]
ctrls_h2  = [LinearLQRController(sub_model),
             LinearLQRController(sub_model),
             LQRController(),
             LQRController(),
             LQRController()]
x0s_h2 = [zeros(state_dim(m)) for m in models_h2]

println("Running H2: 2 subs + 3 quads (pyramid formation)...")
runs_h2 = run_coordinated_sim(models_h2, ctrls_h2, edges_h2, offsets_h2;
                               x0s=x0s_h2, t_end=60.0)
err_h2 = norm(D_h2 * vcat([runs_h2[i].x[1:3, end] for i in 1:n_h2]...) - b_h2)
println("  Formation error at t = 60 s: ", round(err_h2, sigdigits=4), " m")

# ── Plotting functions ────────────────────────────────────────────────────────

"""
    plot_heterogeneous_trajectories(runs, sub_ids, quad_ids)

2x2 grid: rows = x, z; left = submarines, right = quadrotors.
Solid lines are actual trajectories; dashed lines are sheaf references.
"""
function plot_heterogeneous_trajectories(
    runs::Vector,
    sub_ids::Vector{Int},
    quad_ids::Vector{Int},
)
    t = runs[1].t
    sub_cols  = _sub_palette[1:length(sub_ids)]
    quad_cols = _quad_palette[1:length(quad_ids)]

    p = plot(layout=(2, 2), size=(900, 720), link=:x,
             plot_title="Heterogeneous formation trajectories",
             fontfamily="Computer Modern", thickness_scaling=1.5,
             left_margin=12Plots.mm, bottom_margin=8Plots.mm,
             top_margin=4Plots.mm, right_margin=4Plots.mm)

    for (row, (lbl, idx)) in enumerate([("x [m]", 1), ("z [m]", 3)])
        sp_sub  = (row - 1) * 2 + 1
        sp_quad = (row - 1) * 2 + 2

        plot!(p[sp_sub];  ylabel=lbl, xlabel=row==2 ? "t [s]" : "",
              title=row==1 ? "\nSubmarines"  : "", legend=row==1 ? :right : false,
              guidefontsize=10, tickfontsize=8)
        plot!(p[sp_quad]; ylabel=lbl, xlabel=row==2 ? "t [s]" : "",
              title=row==1 ? "\nQuadrotors" : "", legend=row==1 ? :right : false,
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
    plot_xz_paths(runs, edges, sub_ids, quad_ids)

x-z plane view: full trajectory paths, formation edges at final time,
start (hollow) and end (filled) markers, surface line at z = 0.
"""
function plot_xz_paths(
    runs::Vector,
    edges::Vector{Tuple{Int,Int}},
    sub_ids::Vector{Int},
    quad_ids::Vector{Int},
)
    sub_cols  = _sub_palette[1:length(sub_ids)]
    quad_cols = _quad_palette[1:length(quad_ids)]

    all_z = vcat([runs[i].x[3, :] for i in eachindex(runs)]...)
    z_lo  = minimum(all_z) - 0.4
    z_hi  = maximum(all_z) + 0.4

    p = plot(size=(650, 420), xlabel="x [m]", ylabel="z [m]",
             title="Formation paths (x-z plane)",
             legend=:outerright, ylims=(z_lo, z_hi),
             fontfamily="Computer Modern", thickness_scaling=1.5,
             left_margin=6Plots.mm, right_margin=2Plots.mm,
             bottom_margin=6Plots.mm, top_margin=4Plots.mm)

    hline!(p, [0.0]; linestyle=:dash, color=_pp_blue, alpha=0.4,
           label="surface (z = 0)", linewidth=1.2)

    for (j, id) in enumerate(sub_ids)
        plot!(p, runs[id].x[1, :], runs[id].x[3, :];
            color=sub_cols[j], linewidth=2.0, label="Sub $id")
    end
    for (j, id) in enumerate(quad_ids)
        plot!(p, runs[id].x[1, :], runs[id].x[3, :];
            color=quad_cols[j], linewidth=2.0, label="Quad $id")
    end

    pos_end   = [runs[i].x[1:3, end] for i in eachindex(runs)]
    pos_start = [runs[i].x[1:3, 1]   for i in eachindex(runs)]

    for (v1, v2) in edges
        plot!(p, [pos_end[v1][1], pos_end[v2][1]],
                 [pos_end[v1][3], pos_end[v2][3]];
              color=:gray, linewidth=1, linestyle=:dot, label="")
    end

    for (j, id) in enumerate(sub_ids)
        scatter!(p, [pos_start[id][1]], [pos_start[id][3]];
            marker=:dtriangle, markersize=8, color=sub_cols[j],
            markeralpha=0.0, markerstrokewidth=2, markerstrokecolor=sub_cols[j], label="")
        scatter!(p, [pos_end[id][1]], [pos_end[id][3]];
            marker=:dtriangle, markersize=10, color=sub_cols[j], label="")
    end
    for (j, id) in enumerate(quad_ids)
        scatter!(p, [pos_start[id][1]], [pos_start[id][3]];
            marker=:utriangle, markersize=8, color=quad_cols[j],
            markeralpha=0.0, markerstrokewidth=2, markerstrokecolor=quad_cols[j], label="")
        scatter!(p, [pos_end[id][1]], [pos_end[id][3]];
            marker=:utriangle, markersize=10, color=quad_cols[j], label="")
    end

    return p
end

# ── H3: convoy — subs move right, quads follow above ─────────────────────────
#
# Same rectangular formation as H1 (2 subs + 2 quads), but the submarines
# cruise right at 0.1 m/s. The sheaf planner continuously projects the desired
# sub trajectory onto a formation-consistent section, so the quad references
# chase the subs automatically.
#
# Sub 1 leads; sub 2 trails 2 m behind.  Both rise at z = 0.
# Quads hover 3 m above their paired sub throughout the run.

v_convoy = 0.1   # m/s — convoy cruise speed

# sub_traj(i, t) returns the desired 3-D position for sub i at time t
sub_traj_h3 = (i, t) -> i == 1 ? [v_convoy * t,        0.0, 0.0] :
                                   [v_convoy * t - 2.0,  0.0, 0.0]

sub_ids_h3  = [1, 2]
quad_ids_h3 = [3, 4]
n_h3 = 4

# Reuse H1 edges/offsets — same formation geometry, just moving
D_h3, b_h3 = formation_coboundary(n_h3, edges_h1, offsets_h1)

models_h3 = [sub_model, sub_model, quad_model, quad_model]
ctrls_h3  = [LinearLQRController(sub_model),
             LinearLQRController(sub_model),
             LQRController(),
             LQRController()]
x0s_h3 = [zeros(state_dim(m)) for m in models_h3]

println("Running H3: convoy (subs cruise right at $(v_convoy) m/s, quads follow)...")
runs_h3 = run_convoy_sim(models_h3, ctrls_h3, edges_h1, offsets_h1;
                          n_subs=2, sub_traj=sub_traj_h3,
                          x0s=x0s_h3, t_end=60.0)
err_h3 = norm(D_h3 * vcat([runs_h3[i].x[1:3, end] for i in 1:n_h3]...) - b_h3)
println("  Formation error at t = 60 s: ", round(err_h3, sigdigits=4), " m")

# ── Save plots ────────────────────────────────────────────────────────────────

date = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

fp_h1 = "examples/heterogeneous-lqr/figures/h1/"
fp_h2 = "examples/heterogeneous-lqr/figures/h2/"
isdir(fp_h1) || mkdir(fp_h1)
isdir(fp_h2) || mkdir(fp_h2)

println("\nSaving H1 plots...")
savefig(plot_heterogeneous_trajectories(runs_h1, sub_ids_h1, quad_ids_h1),
        fp_h1 * "trajectories_" * date * ".png")
savefig(plot_xz_paths(runs_h1, edges_h1, sub_ids_h1, quad_ids_h1),
        fp_h1 * "paths_xz_"    * date * ".png")
savefig(plot_formation_error(runs_h1, D_h1, b_h1; title="Rectangular formation error"),
        fp_h1 * "formation_error_" * date * ".png")

println("Saving H2 plots...")
savefig(plot_heterogeneous_trajectories(runs_h2, sub_ids_h2, quad_ids_h2),
        fp_h2 * "trajectories_" * date * ".png")
savefig(plot_xz_paths(runs_h2, edges_h2, sub_ids_h2, quad_ids_h2),
        fp_h2 * "paths_xz_"    * date * ".png")
savefig(plot_formation_error(runs_h2, D_h2, b_h2; title="Triangular formation error"),
        fp_h2 * "formation_error_" * date * ".png")

fp_h3 = "examples/heterogeneous-lqr/figures/h3/"
isdir(fp_h3) || mkdir(fp_h3)

println("Saving H3 plots...")
savefig(plot_heterogeneous_trajectories(runs_h3, sub_ids_h3, quad_ids_h3),
        fp_h3 * "trajectories_" * date * ".png")
savefig(plot_xz_paths(runs_h3, edges_h1, sub_ids_h3, quad_ids_h3),
        fp_h3 * "paths_xz_"    * date * ".png")
savefig(plot_formation_error(runs_h3, D_h3, b_h3; title="Convoy formation error"),
        fp_h3 * "formation_error_" * date * ".png")

println("Done. H1 → ", fp_h1, "  H2 → ", fp_h2, "  H3 → ", fp_h3)
