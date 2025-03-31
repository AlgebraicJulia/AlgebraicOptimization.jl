module PaperPlotting

export paper_plot_save_results, postprocess_trajectory, save_trajectories, load_trajectory,
    empty_experiment_plot, plot_trajectory!, blue, orange, green, beige, purple, add_triangle!

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

"""     paper_plot_save_results(trajectory, C, type_str, test_case, additonal_str="", follow_leader=false)

plots the trajectories of each 3 agents and saves the plot and the trajectories as csv files
"""
function paper_plot_save_results(trajectory, C, type_str, test_case, additonal_str="", follow_leader=false)

    # plot
    agent_1_trajectory = mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(1)] for x in trajectory])
    agent_2_trajectory = mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(2)] for x in trajectory])
    agent_3_trajectory = mapreduce(permutedims, vcat, [C * x[BlockArrays.Block(3)] for x in trajectory])

    p = plot(agent_1_trajectory[:, 1], agent_1_trajectory[:, 2], labels="", color=:red)
    if follow_leader
        scatter!(agent_1_trajectory[2:end, 1], agent_1_trajectory[2:end, 2], label="Agent 1 (Leader)", color=:red)
    else
        scatter!(agent_1_trajectory[2:end, 1], agent_1_trajectory[2:end, 2], label="Agent 1", color=:red)
    end
    scatter!([agent_1_trajectory[1, 1]], [agent_1_trajectory[1, 2]], label="", color=:cyan)

    plot!(agent_2_trajectory[:, 1], agent_2_trajectory[:, 2], labels="", color=:blue)
    scatter!(agent_2_trajectory[2:end, 1], agent_2_trajectory[2:end, 2], label="Agent 2", color=:blue)
    scatter!([agent_2_trajectory[1, 1]], [agent_2_trajectory[1, 2]], label="", color=:cyan)

    plot!(agent_3_trajectory[:, 1], agent_3_trajectory[:, 2], labels="", color=:green)
    scatter!(agent_3_trajectory[2:end, 1], agent_3_trajectory[2:end, 2], label="Agent 3", color=:green)
    scatter!([agent_3_trajectory[1, 1]], [agent_3_trajectory[1, 2]], label="Initial Positions", color=:cyan)

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
    CSV.write("./examples/paper-examples/$(type_str)/$(test_case)/$(type_str)$(test_case)_traj1_$(now).csv", Tables.table(agent_1_trajectory))
    CSV.write("./examples/paper-examples/$(type_str)/$(test_case)/$(type_str)$(test_case)_traj2_$(now).csv", Tables.table(agent_2_trajectory))
    CSV.write("./examples/paper-examples/$(type_str)/$(test_case)/$(type_str)$(test_case)_traj3_$(now).csv", Tables.table(agent_3_trajectory))

end

end