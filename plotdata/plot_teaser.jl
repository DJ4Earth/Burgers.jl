using Plots
using JLD

default(titlefont = (14, "times"))
color = :heat

vel = load("plotdata/IC.jld", "vel")
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
println("Initial energy: $senergy")
heatmap(vel;xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/ic.pdf")
vel = load("plotdata/final.jld", "vel")
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
println("Final energy: $senergy")
heatmap(vel; xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/final.pdf")
vel = load("plotdata/adjoints.jld", "dvel")
maximum(vel)
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
println("Adjoint energy: $senergy")
vel .= vel .* 1e6
heatmap(vel; xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/dbc.pdf")
heatmap(vel[2:end-1,2:end-1];xlabel = "x", ylabel = "y", c = color)
savefig("plotdata/dic.pdf")