export get_y, get_x, get_l, get_r, get_u, get_d

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
