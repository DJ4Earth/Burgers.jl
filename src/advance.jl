function advance!(pde::DistPDE{PT}) where {PT}
    stencil!(pde)
    set_boundary_conditions!(pde)
    return nothing
end
