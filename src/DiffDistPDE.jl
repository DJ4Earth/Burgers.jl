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
    export set_initial_conditions!, set_boundary_conditions!, advance!, halo!, get_gx, get_gy, copy_fields!

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

    function set_initial_conditions!(pde::AbstractPDE)
        error("set_initial_conditions! not implemented for $(typeof(pde))")
    end

    function set_boundary_conditions!(pde::AbstractPDE)
        error("set_boundary_conditions! not implemented for $(typeof(pde))")
    end

    function advance!(pde::AbstractPDE)
        error("advance! not implemented for $(typeof(pde))")
    end

    function halo!(pde)
        error("halo! not implemented for $(typeof(pde))")
    end

    function stencil!(pde)
        error("stencil! not implemented for $(typeof(pde))")
    end

    function get_gx(pde::AbstractPDE, i::Int64)
        error("get_gx not implemented for $(typeof(pde))")
    end

    function get_gy(pde::AbstractPDE, i::Int64)
        error("get_gy not implemented for $(typeof(pde))")
    end

    function copy_fields!(dest::AbstractPDE, src::AbstractPDE)
        error("copy_fields! not implemented for $(typeof(src))")
    end

    include("Burgers.jl")
    include("utils.jl")
    include("energy.jl")
    include("dreduction.jl")
    include("update_bc.jl")
end
