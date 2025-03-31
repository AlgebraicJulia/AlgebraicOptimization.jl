include("PaperPlotting.jl")
using .PaperPlotting

using AlgebraicOptimization
using Plots

#=
t1 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/4/velocity-consensus_traj1.csv")
t2 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/4/velocity-consensus_traj2.csv")
t3 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/4/velocity-consensus_traj3.csv")

plt = PaperPlotting.empty_experiment_plot("")

PaperPlotting.plot_trajectory!(plt, t1, "Agent 1", :hexagon, PaperPlotting.orange)
PaperPlotting.plot_trajectory!(plt, t2, "Agent 2", :circle, PaperPlotting.blue)
PaperPlotting.plot_trajectory!(plt, t3, "Agent 3", :diamond, PaperPlotting.green)

savefig(plt, "examples/paper-examples/figures/flocking.png")=#

t1 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/6/flocking6_traj1_2025-03-31T001332191.csv")
t2 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/6/flocking6_traj2_2025-03-31T001332191.csv")
t3 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/6/flocking6_traj3_2025-03-31T001332191.csv")

plt = PaperPlotting.empty_experiment_plot("")

PaperPlotting.plot_trajectory!(plt, t1[1:65, :], "", :hexagon, PaperPlotting.orange)
PaperPlotting.plot_trajectory!(plt, t2[1:65, :], "", :circle, PaperPlotting.blue)
PaperPlotting.plot_trajectory!(plt, t3[1:65, :], "", :diamond, PaperPlotting.green)

savefig(plt, "examples/paper-examples/figures/flocking.png")


# Make flocking plot
t1 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/3/flocking_traj1.csv")
t2 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/3/flocking_traj2.csv")
t3 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/3/flocking_traj3.csv")

plt = PaperPlotting.empty_experiment_plot("")

PaperPlotting.add_triangle!(plt, t1[1, :], t2[1, :], t3[1, :], PaperPlotting.purple)
PaperPlotting.add_triangle!(plt, t1[77, :], t2[77, :], t3[77, :], PaperPlotting.purple)
PaperPlotting.add_triangle!(plt, t1[end, :], t2[end, :], t3[end, :], PaperPlotting.purple)

PaperPlotting.plot_trajectory!(plt, t1, "", :hexagon, PaperPlotting.orange)
PaperPlotting.plot_trajectory!(plt, t2, "", :circle, PaperPlotting.blue)
PaperPlotting.plot_trajectory!(plt, t3, "", :diamond, PaperPlotting.green)

savefig(plt, "examples/paper-examples/figures/moving-formation.png")



# Formation Example

t1 = PaperPlotting.load_trajectory("examples/paper-examples/formation/1/formation1_traj1_2025-03-25T174630847.csv")
t2 = PaperPlotting.load_trajectory("examples/paper-examples/formation/1/formation1_traj2_2025-03-25T174630847.csv")
t3 = PaperPlotting.load_trajectory("examples/paper-examples/formation/1/formation1_traj3_2025-03-25T174630847.csv")

plt = PaperPlotting.empty_experiment_plot("")

# PaperPlotting.add_triangle!(plt, t1[1, :], t2[1, :], t3[1, :], PaperPlotting.purple)
# PaperPlotting.add_triangle!(plt, t1[49, :], t2[49, :], t3[49, :], PaperPlotting.purple)
PaperPlotting.add_triangle!(plt, t1[end, :], t2[end, :], t3[end, :], PaperPlotting.purple)

PaperPlotting.plot_trajectory!(plt, t1, "", :hexagon, PaperPlotting.orange)
PaperPlotting.plot_trajectory!(plt, t2, "", :circle, PaperPlotting.blue)
PaperPlotting.plot_trajectory!(plt, t3, "", :diamond, PaperPlotting.green)

savefig(plt, "examples/paper-examples/figures/stationary-formation.png")

# Consensus Example



t1 = PaperPlotting.load_trajectory("examples/paper-examples/consensus/2/consensus2_traj1_2025-03-28T155604215.csv")
t2 = PaperPlotting.load_trajectory("examples/paper-examples/consensus/2/consensus2_traj2_2025-03-28T155604215.csv")
t3 = PaperPlotting.load_trajectory("examples/paper-examples/consensus/2/consensus2_traj3_2025-03-28T155604215.csv")

plt = PaperPlotting.empty_experiment_plot("")

# PaperPlotting.add_triangle!(plt, t1[1, :], t2[1, :], t3[1, :], PaperPlotting.purple)
# PaperPlotting.add_triangle!(plt, t1[49, :], t2[49, :], t3[49, :], PaperPlotting.purple)
# PaperPlotting.add_triangle!(plt, t1[end, :], t2[end, :], t3[end, :], PaperPlotting.purple)

PaperPlotting.plot_trajectory!(plt, t1, "Agent 1", :hexagon, PaperPlotting.orange)
PaperPlotting.plot_trajectory!(plt, t2, "Agent 2", :circle, PaperPlotting.blue)
PaperPlotting.plot_trajectory!(plt, t3, "Agent 3", :diamond, PaperPlotting.green)

savefig(plt, "examples/paper-examples/figures/tracking-consensus.png")




