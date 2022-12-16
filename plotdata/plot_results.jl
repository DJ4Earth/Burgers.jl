using Plots
using JLD
using Checkpointing

# RAM
steps = 10000
checkpoints = 3
verbose=1
# totalsteps = []
# x = []
# adjoint_factor = 3
# slowdown = []
# for i in 250:10000
#     checkpoints = i
#     revolve = Revolve{Nothing}(steps, checkpoints; verbose=verbose)
#     predfwdcnt = Checkpointing.forwardcount(revolve)
#     push!(x,i)
#     push!(totalsteps, predfwdcnt + adjoint_factor*steps)
#     push!(slowdown, (predfwdcnt + adjoint_factor*steps)/steps)
# end

# plot(x, slowdown;xlim = (250, 10000), ylabel="Slowdown", xlabel="Checkpoints", label="Theoretical runtime")
# savefig("plotdata/results_large.pdf")

tsteps = 10000
totalsteps = []
x = []
adjoint_factor = 3
slowdown = []
reduction = []
for i in 5:250
    checkpoints = i
    revolve = Revolve{Nothing}(steps, checkpoints; verbose=verbose)
    predfwdcnt = Checkpointing.forwardcount(revolve)
    push!(x,i)
    push!(totalsteps, predfwdcnt + adjoint_factor*steps)
    push!(slowdown, (predfwdcnt + adjoint_factor*steps)/steps)
    push!(reduction, tsteps/i)
end
plot(x, slowdown;xlim = (5, 250), ylabel="Adjoint Runtime Overhead", xlabel="Checkpoints", label="Theoretical", legend=:top)
ramresults = [[10, 151.656621,2194.484480], [12, 150.643386 , 2093.868116],[15, 151.315585, 2012.807734], [20, 150.507251,1902.022155], [30, 155.724570, 1842.356589], [50,151.214323, 1741.036194], [100, 152.067012,1715.512279], [150, 152.821876,1574.176904], [200, 151.653466,1690.274661], [250, 152.0, 1676.0]]
resx = []
resslowdown = []
for res in ramresults
 push!(resx, res[1])
 push!(resslowdown, res[3]/res[2])
end
plot!(resx, resslowdown; st=:scatter, label="RAM")
diskresults = [[25, 152.449632, 2482.313448], [50,152.0, 2631.972317], [100, 150.127117, 2102.227693], [150, 151.965112,2287.072575], [200, 151.045504, 2256.360471], [250, 150.0, 3049.214615]]
resx = []
resslowdown = []
for res in diskresults
 push!(resx, res[1])
 push!(resslowdown, res[3]/res[2])
end
plot!(resx, resslowdown; st=:scatter, label="Disk", markershape=:rect)
ssdresults = [[22, 150.803180 , 2349.845388], [25, 152.037776, 2291.132130], [50, 150.704588, 2290.717920], [100, 152.029083, 2037.768621], [150, 150.249174, 2186.939643], [200, 151.762183, 2169.584603 ]]
resx = []
resslowdown = []
for res in ssdresults
 push!(resx, res[1])
 push!(resslowdown, res[3]/res[2])
end
plot!(resx, resslowdown; st=:scatter, label="Node SSD", markershape=:utriangle)
savefig("plotdata/results_runtime.pdf")

plot(x, reduction;xlim = (5, 250), ylabel="Adjoint Memory Reduction", xlabel="Checkpoints", label="Total Checkpoint Memory", legend=:top)
resx = []
resreduction = []
for res in ramresults
 push!(resx, res[1])
 push!(resreduction, tsteps/res[1])
end
# resx = []
# resslowdown = []
# plot!(resx, reduction; st=:scatter, label="RAM")
# for res in diskresults
#  push!(resx, res[1])
#  push!(resreduction, 0)
# end
# plot!(resx, resreduction; st=:scatter, label="Disk")
plot!(resx, resreduction; st=:scatter, label="RAM")
savefig("plotdata/results_memory.pdf")

