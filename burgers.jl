include("header.jl")
include("utils.jl")
include("advance.jl")
include("halo.jl")
include("energy.jl")
include("dreduction.jl")
using Profile
using PProf
using JLD
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
            lastu[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
            lastv[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
        end
        @inbounds for j in 1:size(lastu, 2)
            nextu[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
            nextv[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
        end
    end
    return nothing
end

# Create object from struct.
function main(
    Nx::Int64, Ny::Int64, tsteps::Int64,
    μ::Float64, dx::Float64, dy::Float64,
    dt::Float64, snaps::Int64;
    profile::Bool=false, writedata::Bool=false,
    storage=ArrayStorage{Burgers}(snaps)
)
    burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)

    # Boundary conditions
    set_initial_conditions!(burgers)
    set_boundary_conditions!(burgers)

    vel = (
        burgers.nextu[2:end-1,2:end-1].^2 +
        burgers.nextv[2:end-1,2:end-1].^2
    )
    if writedata
        save("IC.jld", "vel", vel, "u", burgers.nextu, "v", burgers.nextv)
    end

    # heatmap(vel)
    ienergy = energy(burgers)

    np = MPI.Comm_size(burgers.comm)
    rank = MPI.Comm_rank(burgers.comm)
    if burgers.rank == 0
        println("[$rank] Initial energy E = $ienergy")
    end
    # Profile.Allocs.clear()
    # Profile.Allocs.@profile begin
    # Profile.clear()
    # Profile.@profile begin
    @time begin
        set_boundary_conditions!(burgers)
        set_initial_conditions!(burgers)
        global fenergy = final_energy(burgers)
    end
    if burgers.rank == 0
        println("[$rank] Final energy E = $fenergy")
    end
    # PProf.Allocs.pprof()
    # if profile
    # PProf.pprof()
    # end

    vel = (
        burgers.nextu[2:end-1,2:end-1].^2 +
        burgers.nextv[2:end-1,2:end-1].^2
    )
    if writedata
        save("final.jld", "vel", vel, "u", burgers.nextu, "v", burgers.nextv)
    end
    # heatmap(vel)

    burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)
    set_boundary_conditions!(burgers)
    set_initial_conditions!(burgers)
    revolve = Revolve{Burgers}(tsteps, snaps; verbose=1, storage=storage)
    # dburgers = Burgers(Nx, Ny, μ, ν, tsteps)

    @time begin
        set_boundary_conditions!(burgers)
        set_initial_conditions!(burgers)
        Checkpointing.reset(revolve)
        dburgers = Zygote.gradient(final_energy, burgers, revolve)
    end
    # autodiff(final_energy, Active, Duplicated(burgers, dburgers))

    vel = (
        dburgers[1].lastu.^2 +
        dburgers[1].lastv.^2
    )
    if burgers.rank == 0
        println("Norm of energy with respect to initial velocity norm(dE/dv0) = $(norm(vel))")
    end
    if writedata
        save("adjoints.jld", "dvel", vel, "du", dburgers[1].lastu, "dv", dburgers[1].lastv)
    end
    # heatmap(vel)
    # heatmap(dburgers[1].lastu[2:end-1,2:end-1])

end
