# Generate data for the teaser plot. This uses ~105GB of RAM memory for checkpointing
# Run with 1 process. Takes around a total of ~40min to run.

include("burgers.jl")

MPI.Init()
scaling = 100

Nx = 100*scaling
Ny = 100*scaling
# tsteps = 1000*scaling
tsteps = 10000

μ = 0.01 # # U * L / Re,   nu

dx = 3e-2
dy = 3e-2
dt = 1e-3 # dt < 0.5 * dx^2

# snaps = 1000
snaps = 30
rank = MPI.Comm_rank(MPI.COMM_WORLD)
np = MPI.Comm_size(MPI.COMM_WORLD)
if rank == 0
    println("Running Burgers with Nx = $Nx, Ny = $Ny, tsteps = $tsteps, μ = $μ, dx = $dx, dy = $dy, dt = $dt, snaps = $snaps")
    println("with $np processes.")
end
main(Nx, Ny, tsteps, μ, dx, dy, dt, snaps; writedata = false)

if !isinteractive()
    MPI.Finalize()
end