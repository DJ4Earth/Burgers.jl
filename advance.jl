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
