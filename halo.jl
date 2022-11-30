function halo(burgers::Burgers)
    nextu = burgers.nextu
    nextv = burgers.nextv
    rank = burgers.rank
    side = burgers.side
    nx = burgers.nx
    ny = burgers.ny
    comm = burgers.comm
    bufrul = burgers.bufrul
    bufrur = burgers.bufrur
    bufrud = burgers.bufrud
    bufruu = burgers.bufruu
    bufrvl = burgers.bufrvl
    bufrvr = burgers.bufrvr
    bufrvd = burgers.bufrvd
    bufrvu = burgers.bufrvu
    bufsul = burgers.bufsul
    bufsur = burgers.bufsur
    bufsud = burgers.bufsud
    bufsuu = burgers.bufsuu
    bufsvl = burgers.bufsvl
    bufsvr = burgers.bufsvr
    bufsvd = burgers.bufsvd
    bufsvu = burgers.bufsvu

    bufsul .= nextu[2,1:ny]
    bufsur .= nextu[end-1, 1:ny]
    bufsud .= nextu[1:nx,2]
    bufsuu .= nextu[1:nx, end-1]
    bufsvl .= nextv[2,1:ny]
    bufsvr .= nextv[end-1, 1:ny]
    bufsvd .= nextv[1:nx,2]
    bufsvu .= nextv[1:nx, end-1]

    requests = Vector{MPI.Request}()
    # u
    # left right
    if get_x(rank,side) != 0
        req = MPI.Isend(
        bufsul, comm;
        dest=get_l(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrul, comm;
        source=get_l(rank, side), tag=0
        )
        push!(requests, req)
    end
    if get_x(rank,side) != side-1
        req = MPI.Isend(
        bufsur, comm;
        dest=get_r(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrur, comm;
        source=get_r(rank, side), tag=0
        )
        push!(requests, req)
    end
    # up down
    if get_y(rank,side) != 0
        req = MPI.Isend(
        bufsud, comm;
        dest=get_d(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrud, comm;
        source=get_d(rank, side), tag=0
        )
        push!(requests, req)
    end
    if get_y(rank,side) != side-1
        req = MPI.Isend(
        bufsuu, comm;
        dest=get_u(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufruu, comm;
        source=get_u(rank, side), tag=0
        )
        push!(requests, req)
    end
    # v
    # left right
    if get_x(rank,side) != 0
        req = MPI.Isend(
        bufsvl, comm;
        dest=get_l(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrvl, comm;
        source=get_l(rank, side), tag=0
        )
        push!(requests, req)
    end
    if get_x(rank,side) != side-1
        req = MPI.Isend(
        bufsvr, comm;
        dest=get_r(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrvr, comm;
        source=get_r(rank, side), tag=0
        )
        push!(requests, req)
    end
    # up down
    if get_y(rank,side) != 0
        req = MPI.Isend(
        bufsvd, comm;
        dest=get_d(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrvd, comm;
        source=get_d(rank, side), tag=0
        )
        push!(requests, req)
    end
    if get_y(rank,side) != side-1
        req = MPI.Isend(
        bufsvu, comm;
        dest=get_u(rank, side), tag=0
        )
        push!(requests, req)
        req = MPI.Irecv!(
        bufrvu, comm;
        source=get_u(rank, side), tag=0
        )
        push!(requests, req)
    end

    for req in requests
        MPI.Wait(req)
    end

    #u
    if get_x(rank,side) != 0
        for i in 1:ny
            nextu[1,i]  = bufrul[i]
        end
    end
    if get_x(rank,side) != side-1
        for i in 1:ny
            nextu[nx,i] = bufrur[i]
        end
    end
    if get_y(rank,side) != 0
        for i in 1:nx
            nextu[i,1]  = bufrud[i]
        end
    end
    if get_y(rank,side) != side-1
        for i in 1:nx
            nextu[i,ny] = bufruu[i]
        end
    end
    # v
    if get_x(rank,side) != 0
        for i in 1:ny
            nextv[1,i]  = bufrvl[i]
        end
    end
    if get_x(rank,side) != side-1
        for i in 1:ny
            nextv[nx,i] = bufrvr[i]
        end
    end
    if get_y(rank,side) != 0
        for i in 1:nx
            nextv[i,1]  = bufrvd[i]
        end
    end
    if get_y(rank,side) != side-1
        for i in 1:nx
            nextv[i,ny] = bufrvu[i]
        end
    end
    return nothing
end
