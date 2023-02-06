function update_bc!(pde::PDE) where {PDE}
    @unpack lastu, lastv, nextu, nextv, nx, ny, rank, side = pde

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
    @pack! pde =nextu, nextv
end