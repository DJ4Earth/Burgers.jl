module DiffDistPDE
    using LinearAlgebra
    using MPI
    using Plots
    using Checkpointing
    using Zygote
    using Enzyme

    abstract type AbstractPDE end
    export set_initial_conditions!, set_boundary_conditions!, advance!, halo!, get_gx, get_gy, copy_fields!

    function set_initial_conditions!(pde::AbstractPDE)
        error("set_initial_conditions! not implemented for $(typeof(pde))")
    end

    function set_boundary_conditions!(pde::AbstractPDE)
        error("set_boundary_conditions! not implemented for $(typeof(pde))")
    end

    function advance!(pde::AbstractPDE)
        error("advance not implemented for $(typeof(pde))")
    end

    function halo!(pde::AbstractPDE)
        error("halo not implemented for $(typeof(pde))")
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
end
