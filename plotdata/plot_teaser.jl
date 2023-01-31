using Plots
using JLD

# default(titlefont = (14, "times"))
color = :balance

vel = load("IC.jld", "vel")
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
println("Initial energy: $senergy")
surface(
    range(-3, 3, length=size(vel,1)),
    range(-3, 3, length=size(vel,1)),
    vel;xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
savefig("plotdata/ic.pdf")
vel = load("final.jld", "vel")
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
println("Final energy: $senergy")
surface(
    range(-3, 3, length=size(vel,1)),
    range(-3, 3, length=size(vel,1)),
    vel;xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
savefig("plotdata/final.pdf")
vel = load("adjoints.jld", "dvel")
du = load("adjoints.jld", "du")
dv = load("adjoints.jld", "dv")
maximum(vel)
senergy = sum(vel)/(size(vel, 1)*size(vel, 2))
println("Adjoint energy: $senergy")
vel .= vel .* 1e4
surface(
    range(-3, 3, length=size(vel,1)),
    range(-3, 3, length=size(vel,1)),
    vel;xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
savefig("plotdata/dbc.pdf")
surface(
    range(-3, 3, length=size(vel,1)),
    range(-3, 3, length=size(vel,1)),
    du;xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
savefig("plotdata/du.pdf")
surface(
    range(-3, 3, length=size(vel,1)),
    range(-3, 3, length=size(vel,1)),
    dv;xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
savefig("plotdata/dv.pdf")
surface(
    range(-3, 3, length=size(vel,1)),
    range(-3, 3, length=size(vel,1)),
    vel;xlabel = "x", ylabel = "y", c = color
)
savefig("plotdata/dic.pdf")