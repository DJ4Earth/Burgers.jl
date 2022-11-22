include("header.jl")
include("advance.jl")
include("halo.jl")
include("energy.jl")
include("dreduction.jl")

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
    if get_x(rank, side) == 0 && get_y(rank, side) == 0
      lastu[10:20 , 10:20] .= 1.0
      lastv[10:20 , 10:20] .= 1.0
    end
    return nothing
end

MPI.Init()
Nx = 100
Ny = 100
tsteps = 1000
μ = 0.1 # 1/Re
ν = 0.1
# Create object from struct.
burgers = Burgers(Nx, Ny, μ, ν, tsteps)

# Boundary conditions
set_boundary_conditions!(burgers)
set_initial_conditions!(burgers)

vel = sqrt.(
    burgers.lastu.^2 .+
    burgers.lastv.^2
    )
heatmap(vel)

np = MPI.Comm_size(burgers.comm)
rank = MPI.Comm_rank(burgers.comm)
fenergy = final_energy(burgers)
if burgers.rank == 0
  println("[$rank] Final energy E = $fenergy")
end

vel = sqrt.(
    burgers.nextu[2:end-1,2:end-1].^2 +
    burgers.nextv[2:end-1,2:end-1].^2
    )
heatmap(vel)

burgers = Burgers(Nx, Ny, μ, ν, tsteps)
set_boundary_conditions!(burgers)
set_initial_conditions!(burgers)
snaps = 4
revolve = Revolve{Burgers}(tsteps, snaps; verbose=0)
# dburgers = Burgers(Nx, Ny, μ, ν, tsteps)

dburgers = Zygote.gradient(final_energy, burgers, revolve)
# autodiff(final_energy, Active, Duplicated(burgers, dburgers))

vel = sqrt.(
    dburgers[1].lastu[2:end-1,2:end-1].^2 +
    dburgers[1].lastv[2:end-1,2:end-1].^2
    )
vel = sqrt.(
    dburgers[1].lastu[10:20,10:20].^2 +
    dburgers[1].lastv[10:20,10:20].^2
    )
if burgers.rank == 0
  println("Norm of energy with respect to initial velocity norm(dE/dv0) = $(norm(vel))")
end
heatmap(vel)
heatmap(dburgers[1].lastu[2:end-1,2:end-1])

if !isinteractive()
  MPI.Finalize()
end