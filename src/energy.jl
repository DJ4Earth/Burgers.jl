export energy, final_energy

function energy(
    pde::DistPDE{PT},
) where {PT}
    @inbounds lenergy = sum(
        pde.nextu[2:end-1,2:end-1].^2 .+
        pde.nextv[2:end-1,2:end-1].^2
    )
    genergy = MPI.Allreduce(
    lenergy,
    MPI.SUM,
    pde.comm
    )
    return genergy/(pde.Nx * pde.Ny)
end

function final_energy(
    pde::DistPDE{PT},
) where {PT}
    for i in 1:pde.tsteps
        advance!(pde)
        # halo!(pde)
        copyto!(pde.lastu, pde.nextu)
        copyto!(pde.lastv, pde.nextv)
    end
    return energy(pde)
end

function final_energy(
    pde::DistPDE{PT},
    chkpscheme::Scheme,
) where {PT}
    @checkpoint_struct chkpscheme pde for i in 1:pde.tsteps
        advance!(pde)
        # halo!(pde)
        copyto!(pde.lastu, pde.nextu)
        copyto!(pde.lastv, pde.nextv)
    end
    @inbounds lenergy = sum(
    pde.nextu[2:end-1,2:end-1].^2 .+
    pde.nextv[2:end-1,2:end-1].^2
    )
    genergy = lenergy
    # genergy = MPI.Allreduce(
    # lenergy,
    # MPI.SUM,
    # pde.comm
    # )
    return genergy / (pde.Nx * pde.Ny)
end
