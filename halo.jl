get_y(rank::Int, side::Int) = div(rank, side)
get_x(rank::Int, side::Int) = mod(rank, side)
get_l(rank::Int, side::Int) = rank-1
get_r(rank::Int, side::Int) = rank+1
get_u(rank::Int, side::Int) = rank + side
get_d(rank::Int, side::Int) = rank - side

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

  burgers.bufsul .= nextu[2,1:ny]
  burgers.bufsur .= nextu[end-1, 1:ny]
  burgers.bufsud .= nextu[1:nx,2]
  burgers.bufsuu .= nextu[1:nx, end-1]
  burgers.bufsvl .= nextv[2,1:ny]
  burgers.bufsvr .= nextv[end-1, 1:ny]
  burgers.bufsvd .= nextv[1:nx,2]
  burgers.bufsvu .= nextv[1:nx, end-1]

  bufsul = burgers.bufsul
  bufsur = burgers.bufsur
  bufsud = burgers.bufsud
  bufsuu = burgers.bufsuu
  bufsvl = burgers.bufsvl
  bufsvr = burgers.bufsvr
  bufsvd = burgers.bufsvd
  bufsvu = burgers.bufsvu
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

function partition(nx::Int, ny::Int, rank::Int, np::Int)
  side = Int(sqrt(np))
  # Partitioning
  n1x = Int(round(get_x(rank, side) /
                side * (nx+side)))
  n2x = Int(round((get_x(rank, side) + 1) /
                  side * (nx+side)))
  n1y = Int(round(get_y(rank, side) /
                side * (ny+side)))
  n2y = Int(round((get_y(rank, side) + 1) /
                  side * (ny+side)))
  nlx = get_x(rank,side) == 0 ? n1x+1 : n1x
  nrx = get_x(rank,side) == side-1 ? n2x-1 : n2x
  nxlocal = nrx-nlx+1
  nly = get_y(rank,side) == 0 ? n1y+1 : n1y
  nry = get_y(rank,side) == side-1 ? n2y-1 : n2y
  nylocal = nry-nly+1
  return nxlocal, nylocal, side
end