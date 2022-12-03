# Generate data for the teaser plot. This uses ~105GB of RAM memory for checkpointing
# Run with 1 process. Takes around a total of ~40min to run.

include("burgers.jl")

MPI.Init()
scaling = 10

Nx = 100*scaling
Ny = 100*scaling
tsteps = 1000*scaling

μ = 0.01 # # U * L / Re,   nu

dx = 1e-1
dy = 1e-1
dt = 1e-3 # dt < 0.5 * dx^2

snaps = 1000
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    println("Running Burgers with Nx = $Nx, Ny = $Ny, tsteps = $tsteps, μ = $μ, dx = $dx, dy = $dy, dt = $dt, snaps = $snaps")
end
main(Nx, Ny, tsteps, μ, dx, dy, dt, snaps; writedata = true)
# main(Nx, Ny, tsteps, μ, dx, dy, dt)
# main(Nx, Ny, tsteps, μ, dx, dy, dt;profile=true)

if !isinteractive()
    MPI.Finalize()
end