module PaperPlotting

export paper_plot_save_results, postprocess_trajectory, save_trajectories, load_trajectory,
    empty_experiment_plot, plot_trajectory!, blue, orange, green, beige, purple, add_triangle!, plot_trajectories, animate_trajectories, animate_trajectories_save_results

using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
default(fontfamily="Computer Modern")
using CSV, Tables
using Dates

rgb(r, g, b) = RGB(r / 255.0, g / 255.0, b / 255.0)

const blue = rgb(97, 136, 178)
const orange = rgb(223, 167, 119)
const green = rgb(172, 207, 146)
const purple = rgb(216, 201, 238)
const beige = rgb(250, 238, 203)

# Takes the output of do_mpc and postprocesses into matrices of individual agent trajectories
# for saving to CSVs and plotting
function postprocess_trajectory(trajectory, output_maps)
    agent_trajectories = []
    for (i, C) in enumerate(output_maps)
        push!(agent_trajectories,
            mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(i)] for x in trajectory]))
    end
    return agent_trajectories
end

function save_trajectory(filename, trajectory)
    CSV.write(filename, Tables.table(trajectory))
end

function save_trajectories(path, experiment_name, trajectories)
    for (i, t) in enumerate(trajectories)
        f = path * experiment_name * "_traj$(i).csv"
        save_trajectory(f, t)
    end
end

function load_trajectory(trajectory_file)
    return CSV.File(trajectory_file) |> CSV.Tables.matrix
end

function empty_experiment_plot(title; x_label="x", y_label="y")
    plt = plot()
    plot!(plt, title=title, xlabel=x_label, ylabel=y_label, thickness_scaling=1.7)
    return plt
end

function plot_trajectory!(plt, trajectory, label, marker, color)
    plot!(plt, trajectory[:, 1], trajectory[:, 2], label="", lc=color, lw=5)
    scatter!(plt, trajectory[5:4:end, 1], trajectory[5:4:end, 2], label=label, mc=color, markershape=marker, ms=4)
    scatter!(plt, [trajectory[1, 1]], [trajectory[1, 2]], label="", mc=color, markershape=marker, ms=10)
end

function add_triangle!(plt, p1, p2, p3, color)
    plot!(plt, [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]], lc=color, lw=5, label="")
    plot!(plt, [p1[1], p3[1]], [p1[2], p3[2]], lc=color, lw=5, label="")
end

"""
    paper_plot_save_results(trajectory, C, type_str, test_case; additonal_str="", follow_leader=false, n_agents=3)

Plots the trajectories of each agent and saves the plot and the trajectories as csv files.
"""
function paper_plot_save_results(trajectory, C, type_str, test_case; additonal_str="", follow_leader=false, n_agents=3)
    # plot
    agent_trajectories = [mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(i)] for x in trajectory]) for i in 1:n_agents]

    colors = [:red, :blue, :green, :orange, :purple, :black, :magenta, :cyan, :brown, :gray]
    p = plot()
    for i in 1:n_agents
        plot!(p, agent_trajectories[i][:, 1], agent_trajectories[i][:, 2], labels="", color=colors[mod1(i, length(colors))])
        if follow_leader && i == 1
            scatter!(p, agent_trajectories[i][2:end, 1], agent_trajectories[i][2:end, 2], label="Agent $i (Leader)", color=colors[mod1(i, length(colors))])
        else
            scatter!(p, agent_trajectories[i][2:end, 1], agent_trajectories[i][2:end, 2], label="Agent $i", color=colors[mod1(i, length(colors))])
        end
        scatter!(p, [agent_trajectories[i][1, 1]], [agent_trajectories[i][1, 2]], label="", color=:cyan)
    end

    title!("$(type_str) ($(additonal_str))")
    xlabel!("x-position")
    ylabel!("y-position")

    # get current time and remove breaking symbols
    now = Dates.now()
    now = replace(string(now), ":" => "")
    now = replace(string(now), "." => "")

    # example_str to lowercase
    type_str = lowercase(type_str)

    # create directories if they don't exist
    if !isdir("./examples/paper-examples/$(type_str)")
        mkdir("./examples/paper-examples/$(type_str)")
    end
    if !isdir("./examples/paper-examples/$(type_str)/$(test_case)")
        mkdir("./examples/paper-examples/$(type_str)/$(test_case)")
    end

    # save
    savefig(p, "./examples/paper-examples/$(type_str)/$(test_case)/$(type_str)$(test_case)_$(now)")
    for i in 1:n_agents
        CSV.write("./examples/paper-examples/$(type_str)/$(test_case)/$(type_str)$(test_case)_traj$(i)_$(now).csv", Tables.table(agent_trajectories[i]))
    end
end

"""
    plot_trajectories(trajectory, C; triangle=false, moving_triangle=false, n_agents=3)

Plots the trajectories of all agents. Optionally draws triangles for the first three agents.
"""
function plot_trajectories(trajectory, C; triangle=false, moving_triangle=false, n_agents=3)
    # split up trajectories
    agent_trajs = [mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(i)] for x in trajectory]) for i in 1:n_agents]

    plt = PaperPlotting.empty_experiment_plot("")

    # plot triangles (only if at least 3 agents)
    if triangle && n_agents >= 3
        PaperPlotting.add_triangle!(plt, agent_trajs[1][end, :], agent_trajs[2][end, :], agent_trajs[3][end, :], PaperPlotting.purple)
    end

    if moving_triangle && n_agents >= 3
        PaperPlotting.add_triangle!(plt, agent_trajs[1][1, :], agent_trajs[2][1, :], agent_trajs[3][1, :], PaperPlotting.purple)
        PaperPlotting.add_triangle!(plt, agent_trajs[1][77, :], agent_trajs[2][77, :], agent_trajs[3][77, :], PaperPlotting.purple)
        PaperPlotting.add_triangle!(plt, agent_trajs[1][end, :], agent_trajs[2][end, :], agent_trajs[3][end, :], PaperPlotting.purple)
    end

    # plot trajectories
    markers = [:hexagon, :circle, :diamond, :star5, :utriangle, :dtriangle, :rect, :pentagon, :xcross, :vline]
    colors = [PaperPlotting.orange, PaperPlotting.blue, PaperPlotting.green, PaperPlotting.purple, PaperPlotting.beige, :black, :magenta, :cyan, :brown, :gray]
    for i in 1:n_agents
        PaperPlotting.plot_trajectory!(plt, agent_trajs[i], "", markers[mod1(i, length(markers))], colors[mod1(i, length(colors))])
    end

    # show plot
    plot(plt)
end


function compute_axis_limits(agent_trajs; margin=1.0)
    xs = vcat([traj[:, 1] for traj in agent_trajs]...)
    ys = vcat([traj[:, 2] for traj in agent_trajs]...)
    x_min, x_max = minimum(xs), maximum(xs)
    y_min, y_max = minimum(ys), maximum(ys)
    return (x_min - margin, x_max + margin), (y_min - margin, y_max + margin)
end

function animate_trajectories(trajectory, C; n_agents=3, filename="agents.gif", xlims=nothing, ylims=nothing, fps=20)
    colors = [:red, :blue, :green, :orange, :purple, :black, :magenta, :cyan, :brown, :gray]
    agent_trajs = [mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(i)] for x in trajectory]) for i in 1:n_agents]
    # Compute axis limits if not provided
    if xlims === nothing || ylims === nothing
        xlims, ylims = compute_axis_limits(agent_trajs)
    end

    anim = @animate for t in 1:length(trajectory)
        plt = plot(
            title="Agent Consensus Over Time",
            xlabel="x",
            ylabel="y",
            legend=false,
            xlims=xlims,
            ylims=ylims,
        )
        for i in 1:n_agents
            pos = agent_trajs[i][t, :]
            scatter!(
                plt,
                [pos[1]], [pos[2]],
                color=colors[mod1(i, length(colors))],
                ms=5,
            )
        end
    end

    gif(anim, filename, fps=fps)
end

function animate_trajectories_save_results(
    trajectory, C, type_str, test_case;
    additonal_str="", n_agents=3, xlims=nothing, ylims=nothing, fps=20
)
    colors = [:red, :blue, :green, :orange, :purple, :black, :magenta, :cyan, :brown, :gray]
    agent_trajs = [mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(i)] for x in trajectory]) for i in 1:n_agents]
    # Compute axis limits if not provided
    if xlims === nothing || ylims === nothing
        xlims, ylims = compute_axis_limits(agent_trajs)
    end

    anim = @animate for t in 1:length(trajectory)
        plt = plot(
            title="Agent Consensus Over Time",
            xlabel="x",
            ylabel="y",
            legend=false,
            xlims=xlims,
            ylims=ylims,
        )
        for i in 1:n_agents
            pos = agent_trajs[i][t, :]
            scatter!(
                plt,
                [pos[1]], [pos[2]],
                color=colors[mod1(i, length(colors))],
                ms=5,
            )
        end
    end

    # Save in the same directory structure as paper_plot_save_results
    now = Dates.now()
    now = replace(string(now), ":" => "")
    now = replace(string(now), "." => "")
    type_str_lc = lowercase(type_str)
    dir = "./examples/paper-examples/$(type_str_lc)/$(test_case)"
    if !isdir(dir)
        mkpath(dir)
    end
    filename = "$(dir)/$(type_str_lc)$(test_case)_$(additonal_str)_$(now).gif"
    gif(anim, filename, fps=fps)
end

end