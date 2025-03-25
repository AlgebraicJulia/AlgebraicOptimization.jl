module PaperPlotting

export paper_plot_save_results

using Test
using AlgebraicOptimization
using LinearAlgebra
using BlockArrays
using Plots
using CSV, Tables
using Dates

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