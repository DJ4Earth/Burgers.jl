using Plots
using JLD

default(titlefont = (14, "times"))
color = :heat

vel = load("plotdata/IC.jld", "vel")
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
heatmap(vel; title = "Initial velocity vel(u,v) with energy E = $senergy", xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/ic.pdf")
vel = load("plotdata/final.jld", "vel")
senergy = sum(vel)
heatmap(vel; title = "Final velocity vel(u,v) with energy E = $senergy", xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/final.pdf")
vel = load("plotdata/adjoints.jld", "dvel")
senergy = sum(vel)
heatmap(vel; title = "Final velocity with respect to BCs dvel(u,v) with energy dE= $senergy", xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/dbc.pdf")
heatmap(vel[2:end-1,2:end-1]; title = "Final velocity with respect to ICs dvel(u,v) with energy dE = $senergy", xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/dic.pdf")