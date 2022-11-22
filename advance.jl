function advance(burgers::Burgers)
    nextu = burgers.nextu
    nextv = burgers.nextv
    lastu = burgers.lastu
    lastv = burgers.lastv
    μ = burgers.μ
    ν = burgers.ν
    nx = burgers.nx
    ny = burgers.ny
    rank = burgers.rank
    side = burgers.side
    for i in 2:(nx-1)
        for j in 2:(ny-1)
            nextu[i,j] = lastu[i,j] - μ * (
            lastu[i,j]*(lastu[i,j]-lastu[i-1,j]) -
            lastv[i,j]*(lastu[i,j]-lastu[i,j-1])
            ) +
            ν * (
            lastu[i+1,j]-2*lastu[i,j]+ lastu[i-1,j] +
            (lastu[i,j+1]-2*lastu[i,j]+ lastu[i,j-1])
            )
            nextv[i,j] = lastv[i,j]- μ * (
            lastu[i,j]*(lastv[i,j]-lastv[i-1,j]) -
            lastv[i,j]*(lastv[i,j]-lastv[i,j-1])
            ) +
            ν * (
            (lastv[i+1,j]-2*lastv[i,j]+ lastv[i-1,j]) +
            (lastv[i,j+1]-2*lastv[i,j]+ lastv[i,j-1])
            )
        end
    end
    if get_x(rank, side) == 0
        for i in 1:ny
            nextu[1,i] = lastu[1,i]
            nextv[1,i] = lastv[1,i]
        end
    end
    if get_x(rank, side) == side-1
        for i in 1:ny
            nextu[nx,i] = lastu[nx,i]
            nextv[nx,i] = lastv[nx,i]
        end
    end
    if get_y(rank, side) == 0
        for i in 1:nx
            nextu[i,1] = lastu[i,1]
            nextv[i,1] = lastv[i,1]
        end
    end
    if get_y(rank, side) == side-1
        for i in 1:nx
            nextu[i,ny] = lastu[i,ny]
            nextv[i,ny] = lastv[i,ny]
        end
    end
    return nothing
end
