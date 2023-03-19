using Checkpointing
using DiffDistPDE
using JLD
using LinearAlgebra
using MPI
using PProf
using Profile
using Test
using Zygote

include("../examples/burgers.jl")

@testset "Testing adjoint Burgers with Gaussian intiatial conditions" begin

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
    ienergy, fenergy = burgers(Nx, Ny, tsteps, μ, dx, dy, dt)
    dlastu = Float64[]
    for h in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        hburgers = DistPDE{Burgers}(Nx, Ny, μ, dx, dy, dt, tsteps)
        set_initial_conditions!(hburgers)
        set_boundary_conditions!(hburgers)
        hburgers.lastu[55,46] += h
        push!(dlastu, (final_energy(hburgers) - fenergy)/h)
    end
    ndvel, dburgers = burgers_adjoint(Nx, Ny, tsteps, μ, dx, dy, dt, snaps)
    @test ienergy ≈ 0.0855298595153226
    @test fenergy ≈ 0.08426001732938161
    @test ndvel ≈ 1.3020729832060115e-6
    @test isapprox(dlastu[2], dburgers[1].lastu[55,46], atol=1e-8)
end
