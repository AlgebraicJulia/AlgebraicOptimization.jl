include("PaperPlotting.jl")
using .PaperPlotting

using AlgebraicOptimization

# Make flocking plot
t1 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/1/flocking_traj1.csv")
t2 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/1/flocking_traj2.csv")
t3 = PaperPlotting.load_trajectory("examples/paper-examples/flocking/1/flocking_traj3.csv")

plt = PaperPlotting.empty_experiment_plot("Flocking")

PaperPlotting.add_triangle!(plt, t1[1, :], t2[1, :], t3[1, :], PaperPlotting.purple)
PaperPlotting.add_triangle!(plt, t1[49, :], t2[49, :], t3[49, :], PaperPlotting.purple)
PaperPlotting.add_triangle!(plt, t1[end, :], t2[end, :], t3[end, :], PaperPlotting.purple)

PaperPlotting.plot_trajectory!(plt, t1, "Agent 1", :hexagon, PaperPlotting.orange)
PaperPlotting.plot_trajectory!(plt, t2, "Agent 2", :circle, PaperPlotting.blue)
PaperPlotting.plot_trajectory!(plt, t3, "Agent 3", :diamond, PaperPlotting.green)




