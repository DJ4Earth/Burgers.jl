using Checkpointing
using DiffDistPDE
using JLD
using LinearAlgebra
using MPI
using PProf
using Profile
using Test
using Zygote

@testset "Testing adjoint Burgers with Gaussian intiatial conditions" begin

    include("../examples/gaussian_ic.jl")

    MPI.Init()
    scaling = 1

    Nx = 100*scaling
    Ny = 100*scaling
    tsteps = 1000*scaling

    μ = 0.01 # # U * L / Re,   nu

    dx = 1e-1
    dy = 1e-1
    dt = 1e-3 # dt < 0.5 * dx^2

    snaps = 100
    println("Running Burgers with Nx = $Nx, Ny = $Ny, tsteps = $tsteps, μ = $μ, dx = $dx, dy = $dy, dt = $dt, snaps = $snaps")
    ienergy, fenergy, ndvel = burgers(Nx, Ny, tsteps, μ, dx, dy, dt, snaps)
    @test ienergy ≈ 0.0855298595153226
    @test fenergy ≈ 0.08426001732938161
    @test ndvel ≈ 1.3020729832060115e-6

    if !isinteractive()
        MPI.Finalize()
    end
end
