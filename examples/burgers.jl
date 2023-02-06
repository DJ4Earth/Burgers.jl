using Checkpointing
using DiffDistPDE
using JLD
using LinearAlgebra
using MPI
using Parameters
using PProf
using Profile
using Zygote

struct Burgers <: AbstractPDE end

function DiffDistPDE.set_boundary_conditions!(burgers::DistPDE{Burgers})
    @unpack rank, side, lastu, lastv, nx, ny = burgers
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
    @pack! burgers = lastu, lastv
    return nothing
end

function DiffDistPDE.stencil!(burgers::DistPDE{Burgers})
    @unpack lastu, nextu, lastv, nextv, dx, dy, dt, μ, nx, ny = burgers
    @inbounds for i in 2:(nx-1)
        @inbounds for j in 2:(ny-1)
            nextu[i,j] = lastu[i,j] + dt * ( (
            - lastu[i,j]/(2*dx)*(lastu[i+1,j]-lastu[i-1,j])
            - lastv[i,j]/(2*dy)*(lastu[i,j+1]-lastu[i,j-1])
            ) +
            μ * (
            (lastu[i+1,j]-2*lastu[i,j]+ lastu[i-1,j])/dx^2 +
            (lastu[i,j+1]-2*lastu[i,j]+ lastu[i,j-1])/dy^2
            ) )

            nextv[i,j] = lastv[i,j] + dt * ( (
            - lastu[i,j]/(2*dx)*(lastv[i+1,j]-lastv[i-1,j])
            - lastv[i,j]/(2*dy)*(lastv[i,j+1]-lastv[i,j-1])
            ) +
            μ * (
            (lastv[i+1,j]-2*lastv[i,j]+ lastv[i-1,j])/dx^2 +
            (lastv[i,j+1]-2*lastv[i,j]+ lastv[i,j-1])/dy^2
            ) )
        end
    end
    @pack! burgers = nextu, nextv
end

function set_initial_conditions!(burgers::DistPDE{Burgers})
    function get_gx(burgers::DistPDE{Burgers}, x::Int64)
        dx = 6.0 / (burgers.Nx-1)
        return (get_x(burgers.rank, burgers.side) * (burgers.nx-2) + x-1) * dx - 3.0
    end

    function get_gy(burgers::DistPDE{Burgers}, y::Int64)
        dy = 6.0 / (burgers.Ny-1)
        return (get_y(burgers.rank, burgers.side) * (burgers.ny-2) + y-1) * dy - 3.0
    end
    @unpack lastu, lastv, nextu, nextv, nx, ny = burgers
    @inbounds for i in 1:nx
        @inbounds for j in 1:ny
            lastu[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
            lastv[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
            nextu[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
            nextv[i,j] = exp(-get_gx(burgers, i)^2 - get_gy(burgers, j)^2)
        end
    end
    @pack! burgers = lastu, lastv, nextu, nextv
    return nothing
end

# Create object from struct.
function burgers(
    Nx::Int64, Ny::Int64, tsteps::Int64,
    μ::Float64, dx::Float64, dy::Float64,
    dt::Float64, snaps::Int64;
    profile::Bool=false, writedata::Bool=false,
    storage=ArrayStorage{DistPDE{Burgers}}(snaps)
)
    burgers = DistPDE{Burgers}(Nx, Ny, μ, dx, dy, dt, tsteps)

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

    burgers = DistPDE{Burgers}(Nx, Ny, μ, dx, dy, dt, tsteps)
    set_boundary_conditions!(burgers)
    set_initial_conditions!(burgers)
    revolve = Revolve{DistPDE{Burgers}}(tsteps, snaps; verbose=1, storage=storage)

    @time begin
        set_boundary_conditions!(burgers)
        set_initial_conditions!(burgers)
        Checkpointing.reset(revolve)
        dburgers = Zygote.gradient(final_energy, burgers, revolve)
    end

    dvel = (
        dburgers[1].lastu.^2 +
        dburgers[1].lastv.^2
    )
    if burgers.rank == 0
        println("Norm of energy with respect to initial velocity norm(dE/dv0) = $(norm(dvel))")
    end
    if writedata
        save("adjoints.jld", "dvel", vel, "du", dburgers[1].lastu, "dv", dburgers[1].lastv)
    end
    # heatmap(dvel)
    # heatmap(dburgers[1].lastu[2:end-1,2:end-1])
    return ienergy, fenergy, norm(dvel)
end
