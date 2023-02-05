export Burgers

mutable struct Burgers
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

function Burgers(
    Nx::Int,
    Ny::Int,
    μ::Float64,
    dx::Float64,
    dy::Float64,
    dt::Float64,
    tsteps::Int;
    comm::MPI.Comm = MPI.COMM_WORLD
)
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
    return Burgers(
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

function advance(burgers::Burgers)
    nextu = burgers.nextu
    nextv = burgers.nextv
    lastu = burgers.lastu
    lastv = burgers.lastv
    μ = burgers.μ
    # ν = burgers.ν
    nx = burgers.nx
    ny = burgers.ny
    dt = burgers.dt
    dx = burgers.dx
    dy = burgers.dy
    rank = burgers.rank
    side = burgers.side
    @inbounds for i in 2:(nx-1)
        @inbounds for j in 2:(ny-1)

            nextu[i,j] = lastu[i,j] + dt * ( (
            - lastu[i,j]/(2*dx)*(lastu[i+1,j]-lastu[i-1,j])
            - lastv[i,j]/(2*dy)*(lastu[i,j+1]-lastu[i,j-1])
            ) +
            μ * (
            (lastu[i+1,j]-2*lastu[i,j]+ lastu[i-1,j])/dx^2 +
            (lastu[i,j+1]-2*lastu[i,j]+ lastu[i,j-1])/dy^2
            ) )

            nextv[i,j] = lastv[i,j] + dt * ( (
            - lastu[i,j]/(2*dx)*(lastv[i+1,j]-lastv[i-1,j])
            - lastv[i,j]/(2*dy)*(lastv[i,j+1]-lastv[i,j-1])
            ) +
            μ * (
            (lastv[i+1,j]-2*lastv[i,j]+ lastv[i-1,j])/dx^2 +
            (lastv[i,j+1]-2*lastv[i,j]+ lastv[i,j-1])/dy^2
            ) )
        end
    end
    if get_x(rank, side) == 0
        @inbounds for i in 1:ny
            nextu[1,i] = lastu[1,i]
            nextv[1,i] = lastv[1,i]
        end
    end
    if get_x(rank, side) == side-1
        @inbounds for i in 1:ny
            nextu[nx,i] = lastu[nx,i]
            nextv[nx,i] = lastv[nx,i]
        end
    end
    if get_y(rank, side) == 0
        @inbounds for i in 1:nx
            nextu[i,1] = lastu[i,1]
            nextv[i,1] = lastv[i,1]
        end
    end
    if get_y(rank, side) == side-1
        @inbounds for i in 1:nx
            nextu[i,ny] = lastu[i,ny]
            nextv[i,ny] = lastv[i,ny]
        end
    end
    return nothing
end

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
        @inbounds for i in 1:ny
            nextu[1,i]  = bufrul[i]
        end
    end
    if get_x(rank,side) != side-1
        @inbounds for i in 1:ny
            nextu[nx,i] = bufrur[i]
        end
    end
    if get_y(rank,side) != 0
        @inbounds for i in 1:nx
            nextu[i,1]  = bufrud[i]
        end
    end
    if get_y(rank,side) != side-1
        @inbounds for i in 1:nx
            nextu[i,ny] = bufruu[i]
        end
    end
    # v
    if get_x(rank,side) != 0
        @inbounds for i in 1:ny
            nextv[1,i]  = bufrvl[i]
        end
    end
    if get_x(rank,side) != side-1
        @inbounds for i in 1:ny
            nextv[nx,i] = bufrvr[i]
        end
    end
    if get_y(rank,side) != 0
        @inbounds for i in 1:nx
            nextv[i,1]  = bufrvd[i]
        end
    end
    if get_y(rank,side) != side-1
        @inbounds for i in 1:nx
            nextv[i,ny] = bufrvu[i]
        end
    end
    return nothing
end

# get global coordinates

function get_gx(burgers::Burgers, x::Int64)
    dx = 6.0 / (burgers.Nx-1)
    return (get_x(burgers.rank, burgers.side) * (burgers.nx-2) + x-1) * dx - 3.0
end

function get_gy(burgers::Burgers, y::Int64)
    dy = 6.0 / (burgers.Ny-1)
    return (get_y(burgers.rank, burgers.side) * (burgers.ny-2) + y-1) * dy - 3.0
end
