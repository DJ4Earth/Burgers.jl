get_y(rank::Int, side::Int) = div(rank, side)
get_x(rank::Int, side::Int) = mod(rank, side)
get_l(rank::Int, side::Int) = rank-1
get_r(rank::Int, side::Int) = rank+1
get_u(rank::Int, side::Int) = rank + side
get_d(rank::Int, side::Int) = rank - side

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
# get global coordinates

function get_gx(burgers::Burgers, x::Int64)
    dx = 6.0 / (burgers.Nx-1)
    return (get_x(burgers.rank, burgers.side) * (burgers.nx-2) + x-1) * dx - 3.0
end

function get_gy(burgers::Burgers, y::Int64)
    dy = 6.0 / (burgers.Ny-1)
    return (get_y(burgers.rank, burgers.side) * (burgers.ny-2) + y-1) * dy - 3.0
end
