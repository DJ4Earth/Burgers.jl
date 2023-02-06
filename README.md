# DiffDistPDE
[![CI](https://github.com/DJ4Earth/Burgers.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/DJ4Earth/Burgers.jl/actions/workflows/ci.yml)

DiffDistPDE is an implementation of a differentiable distributed PDE solver in Julia. It is developed using the latest language features and packages of Julia, which required ample workarounds and is meant as a demonstration of future capabilities for developing differentiable numerical simulation software. It uses
* [MPI.jl](https://github.com/JuliaParallel/MPI.jl) for distributed computing,
* [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for differentiation,
* and [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) for reducing the memory footprint of the differentiated code.

## Quickstart

Example usage of the solver is given in the [examples](examples) folder. The following example solves the 2D Burgers equation with Dirichlet boundary conditions
```math
u(t, x, −L) = u(t, x, L) = u(t, −L, y) = u(t, L, y) = 0
```
and initial condition
```math
u(0, x, y) = exp(−x^2 − y^2), v(0, x, y) = exp(−x^2 − y^2)
```

```julia
include("examples/burgers.jl")

MPI.Init()
scaling = 10

Nx = 100*scaling
Ny = 100*scaling
tsteps = 1000*scaling

μ = 0.01 # # U * L / Re,   nu

dx = 3e-2
dy = 3e-2
dt = 1e-3 # dt < 0.5 * dx^2

snaps = 1000
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    println(
        "Running Burgers with Nx = $Nx, Ny = $Ny, tsteps = $tsteps,
        μ = $μ, dx = $dx, dy = $dy, dt = $dt, snaps = $snaps"
    )
end
burgers(Nx, Ny, tsteps, μ, dx, dy, dt, snaps; writedata = true)
```
The following files will be written
* `IC.jld` - Initial condition
* `final.jld` - Final state
* `adjoints.fld` - Adjoint state of the final energy with respect to the initial state

The plots can be generated using [plotdata/plot_teaser.jl](plotdata/plot_teaser.jl).

Please note that the solver is not optimized for performance and is not meant for production use. It is meant as a demonstration of Julia for developing differentiable numerical simulation software.