function energy(burgers::Burgers)
    lenergy = sum(
    burgers.nextu[2:end-1,2:end-1].^2 .+
    burgers.nextv[2:end-1,2:end-1].^2
    )
    genergy = MPI.Allreduce(
    lenergy,
    MPI.SUM,
    burgers.comm
    )
    return genergy/(burgers.Nx * burgers.Ny)
end

function final_energy(
    burgers::Burgers,
    )
    for i in 1:burgers.tsteps
        advance(burgers)
        halo(burgers)
        copyto!(burgers.lastu, burgers.nextu)
        copyto!(burgers.lastv, burgers.nextv)
    end
    return energy(burgers)
end

function final_energy(
    burgers::Burgers,
    chkpscheme::Scheme,
    )
    @checkpoint_struct chkpscheme burgers for i in 1:burgers.tsteps
        advance(burgers)
        halo(burgers)
        copyto!(burgers.lastu, burgers.nextu)
        copyto!(burgers.lastv, burgers.nextv)
    end
    lenergy = sum(
    burgers.nextu[2:end-1,2:end-1].^2 .+
    burgers.nextv[2:end-1,2:end-1].^2
    )
    genergy = MPI.Allreduce(
    lenergy,
    MPI.SUM,
    burgers.comm
    )
    return genergy/(float(Nx * Ny))
end
