module DiffDistPDE
    using Checkpointing
    using Enzyme
    using LinearAlgebra
    using MPI
    using Parameters
    using Plots
    using Zygote

    abstract type AbstractPDE end
    export AbstractPDE, DistPDE
    export set_initial_conditions!, set_boundary_conditions!, advance!, halo!, stencil!, get_gx, get_gy

    """
    `DistPDE{PT}` is a distributed PDE solver for a PDE of type `PT`.

    `lastu`, `nextu`, `lastv`, `nextv` are the fields of the PDE.
    `nx` and `ny` are the local grid dimensions.
    `Nx` and `Ny` are the global grid dimensions.
    `μ` is the diffusion coefficient.
    `tsteps` is the number of time steps.
    `np` is the number of processes.
    `side` is the number of processes in each dimension.
    `rank` is the rank of the process.
    `bufxxx` are the buffers for halo exchange.
    """
    @with_kw mutable struct DistPDE{PT <: AbstractPDE}
        nextu::Matrix{Float64}
        nextv::Matrix{Float64}
        lastu::Matrix{Float64}
        lastv::Matrix{Float64}
        nx::Int
        ny::Int
        Nx::Int
        Ny::Int
        μ::Float64
        dx::Float64
        dy::Float64
        dt::Float64
        tsteps::Int
        np::Int
        side::Int
        rank::Int
        comm::MPI.Comm
        # rs send/recv, uv fields u/v, lrud, left right up down
        bufrul::Vector{Float64}
        bufrur::Vector{Float64}
        bufrud::Vector{Float64}
        bufruu::Vector{Float64}

        bufrvl::Vector{Float64}
        bufrvr::Vector{Float64}
        bufrvd::Vector{Float64}
        bufrvu::Vector{Float64}

        bufsul::Vector{Float64}
        bufsur::Vector{Float64}
        bufsud::Vector{Float64}
        bufsuu::Vector{Float64}

        bufsvl::Vector{Float64}
        bufsvr::Vector{Float64}
        bufsvd::Vector{Float64}
        bufsvu::Vector{Float64}
    end

"""
    `DistPDE{PT}(
        Nx::Int, Ny::Int,
        μ::Float64,
        dx::Float64, dy::Float64,
        dt::Float64, tsteps::Int;
        comm::MPI.Comm = MPI.COMM_WORLD)`

    Constructs a distributed PDE solver of type `PT` with `Nx` and `Ny` grid
    points, `μ` diffusion coefficient, `dx` and `dy` grid spacings, `dt` time
    step, `tsteps` number of time steps, and `comm` MPI communicator.
    """
    function DistPDE{PT}(
        Nx::Int,
        Ny::Int,
        μ::Float64,
        dx::Float64,
        dy::Float64,
        dt::Float64,
        tsteps::Int;
        comm::MPI.Comm = MPI.COMM_WORLD
    ) where {PT <: AbstractPDE}
        comm = MPI.COMM_WORLD
        np = MPI.Comm_size(comm)
        rank = MPI.Comm_rank(comm)
        nxlocal, nylocal, side = partition(Nx, Ny, rank, np)
        nextu = zeros(nxlocal, nylocal)
        nextv = zeros(nxlocal, nylocal)
        lastu = zeros(nxlocal, nylocal)
        lastv = zeros(nxlocal, nylocal)
        bufrul = zeros(nylocal)
        bufrur = zeros(nylocal)
        bufrud = zeros(nxlocal)
        bufruu = zeros(nxlocal)
        bufrvl = zeros(nylocal)
        bufrvr = zeros(nylocal)
        bufrvd = zeros(nxlocal)
        bufrvu = zeros(nxlocal)
        bufsul = zeros(nylocal)
        bufsur = zeros(nylocal)
        bufsud = zeros(nxlocal)
        bufsuu = zeros(nxlocal)
        bufsvl = zeros(nylocal)
        bufsvr = zeros(nylocal)
        bufsvd = zeros(nxlocal)
        bufsvu = zeros(nxlocal)
        return DistPDE{PT}(
            nextu, nextv, lastu, lastv,
            nxlocal, nylocal, Nx, Ny,
            μ,
            dx, dy, dt,
            tsteps, np, side,
            rank, comm,
            bufrul, bufrur, bufrud, bufruu,
            bufrvl, bufrvr, bufrvd, bufrvu,
            bufsul, bufsur, bufsud, bufsuu,
            bufsvl, bufsvr, bufsvd, bufsvu,
        )
    end

    """
        set_initial_conditions!(pde::DistPDE{AbstractPDE})

    Set the initial conditions for the PDE.
    """
    function set_initial_conditions!(pde::DistPDE{AbstractPDE})
        error("set_initial_conditions! not implemented for $(typeof(pde))")
    end

    """
        set_boundary_conditions!(pde::DistPDE{AbstractPDE})

    Set the boundary conditions for the PDE.
    """
    function set_boundary_conditions!(pde::DistPDE{AbstractPDE})
        error("set_boundary_conditions! not implemented for $(typeof(pde))")
    end

    """
        advance!(pde::DistPDE{AbstractPDE})

    Advance the PDE by one time step.
    """
    function advance!(pde::DistPDE{AbstractPDE})
        error("advance! not implemented for $(typeof(pde))")
    end

    """
        halo!(pde::DistPDE{AbstractPDE})

    Exchange halo regions between neighboring processes.
    """
    function halo!(pde::DistPDE{AbstractPDE})
        error("halo! not implemented for $(typeof(pde))")
    end

    """
        stencil!(pde::DistPDE{AbstractPDE})

    Apply the stencil to the PDE.
    """
    function stencil!(pde::DistPDE{AbstractPDE})
        error("stencil! not implemented for $(typeof(pde))")
    end

    """
        get_gx(pde::DistPDE{AbstractPDE}, i::Int64)

    Get the x-value at local index `i`.
    """
    function get_gx(pde::DistPDE{AbstractPDE}, i::Int64)
        error("get_gx not implemented for $(typeof(pde))")
    end

    """
        get_gy(pde::DistPDE{AbstractPDE}, i::Int64)

    Get the y-value at local index `i`.
    """
    function get_gy(pde::DistPDE{AbstractPDE}, i::Int64)
        error("get_gy not implemented for $(typeof(pde))")
    end

    include("advance.jl")
    include("halo.jl")
    include("utils.jl")
    include("energy.jl")
    include("dreduction.jl")
    include("update_bc.jl")
end
