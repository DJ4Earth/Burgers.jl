include("header.jl")
include("utils.jl")
include("advance.jl")
include("halo.jl")
include("energy.jl")
include("dreduction.jl")
using Profile
using PProf
function set_boundary_conditions!(burgers::Burgers)
    rank = burgers.rank
    side = burgers.side
    lastu = burgers.lastu
    lastv = burgers.lastv
    nx = burgers.nx
    ny = burgers.ny
    if get_x(rank, side) == 0
        lastu[1:1,1:ny] .= 0.0
        lastv[1:1,1:ny] .= 0.0
    end
    if get_x(rank, side) == side-1
        lastu[nx:nx,1:ny] .= 0.0
        lastv[nx:nx,1:ny] .= 0.0
    end
    if get_y(rank, side) == 0
        lastu[1:nx,1:1] .= 0.0
        lastv[1:nx,1:1] .= 0.0
    end
    if get_y(rank, side) == side-1
        lastu[1:nx,ny:ny] .= 0.0
        lastv[1:nx,ny:ny] .= 0.0
    end
    return nothing
end

function set_initial_conditions!(burgers::Burgers)
    rank = burgers.rank
    side = burgers.side
    lastu = burgers.lastu
    lastv = burgers.lastv
    nextu = burgers.nextu
    nextv = burgers.nextv
    nx = burgers.nx
    ny = burgers.ny
    @inbounds for i in 1:size(lastu, 1)
        @inbounds for j in 1:size(lastu, 2)
            lastu[i,j] = sin(get_gx(burgers, i))
            lastv[i,j] = sin(get_gy(burgers, j))
        end
        @inbounds for j in 1:size(lastu, 2)
            nextu[i,j] = sin(get_gx(burgers, i))
            nextv[i,j] = sin(get_gy(burgers, j))
        end
    end
    return nothing
end

# Create object from struct.
function main(Nx::Int64, Ny::Int64, tsteps::Int64, μ::Float64, dx::Float64, dy::Float64, dt::Float64; profile::Bool=false)
    burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)

    # Boundary conditions
    set_initial_conditions!(burgers)
    set_boundary_conditions!(burgers)

    vel = sqrt.(
    burgers.nextu[2:end-1,2:end-1].^2 +
    burgers.nextv[2:end-1,2:end-1].^2
    )

    # heatmap(vel)
    ienergy = energy(burgers)

    np = MPI.Comm_size(burgers.comm)
    rank = MPI.Comm_rank(burgers.comm)
    if burgers.rank == 0
        println("[$rank] Initial energy E = $ienergy")
    end
    # Profile.Allocs.clear()
    # Profile.Allocs.@profile begin
    Profile.clear()
    Profile.@profile begin
    global fenergy = final_energy(burgers)
    end
    if burgers.rank == 0
        println("[$rank] Final energy E = $fenergy")
    end
    # PProf.Allocs.pprof()
    if profile
    PProf.pprof()
    end

    vel = sqrt.(
    burgers.nextu[2:end-1,2:end-1].^2 +
    burgers.nextv[2:end-1,2:end-1].^2
    )
    heatmap(vel)

    burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)
    set_boundary_conditions!(burgers)
    set_initial_conditions!(burgers)
    snaps = 100
    revolve = Revolve{Burgers}(tsteps, snaps; verbose=0)
    # dburgers = Burgers(Nx, Ny, μ, ν, tsteps)

    dburgers = Zygote.gradient(final_energy, burgers, revolve)
    # autodiff(final_energy, Active, Duplicated(burgers, dburgers))

    vel = sqrt.(
    dburgers[1].lastu[2:end-1,2:end-1].^2 +
    dburgers[1].lastv[2:end-1,2:end-1].^2
    )
    if burgers.rank == 0
        println("Norm of energy with respect to initial velocity norm(dE/dv0) = $(norm(vel))")
    end
    heatmap(vel)
    # heatmap(dburgers[1].lastu[2:end-1,2:end-1])

end
MPI.Init()
scaling = 10

Nx = 100*scaling
Ny = 100*scaling
tsteps = 1000*scaling

μ = 0.01 # # U * L / Re,   nu

dx = 1e-2
dy = 1e-2
dt = 1e-3
main(Nx, Ny, tsteps, μ, dx, dy, dt)
main(Nx, Ny, tsteps, μ, dx, dy, dt)
main(Nx, Ny, tsteps, μ, dx, dy, dt;profile=true)

if !isinteractive()
    MPI.Finalize()
end
