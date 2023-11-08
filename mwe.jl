using Enzyme
using KernelAbstractions


@kernel function stencil_kernel!(
    nextu, nextv, lastu, lastv,
    @Const(dx), @Const(dy), @Const(dt), @Const(μ),
    @Const(nx), @Const(ny),
    )
    i, j = @index(Global, NTuple)
    if i > 1 && j > 1 && i < nx && j < ny
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

function stencil!(nextu, nextv, lastu, lastv, dx, dy, dt, μ, nx, ny)
    stencil_kernel!(CPU())(nextu, nextv, lastu, lastv, dx, dy, dt, μ, nx, ny; ndrange = (nx, ny))
    synchronize(CPU())
    return nothing
end

function main()
    nx = 10
    ny = 10
    lastu = zeros(nx, ny)
    lastv = zeros(nx, ny)
    lastu[1,:] .= 1.0
    lastu[nx,:] .= 1.0
    lastu[:,1] .= 1.0
    lastu[:,ny] .= 1.0
    nextu = copy(lastu)
    nextv = copy(lastv)
    dx = 0.1
    dy = 0.1
    dt = 0.001
    μ = 0.1
    stencil!(nextu, nextv, lastu, lastv, dx, dy, dt, μ, nx, ny)

    dlastu = copy(lastu)
    dlastv = copy(lastv)
    dnextu = copy(nextu)
    dnextv = copy(nextv)

    Enzyme.autodiff(
        ReverseWithPrimal, stencil!,
        Duplicated(nextu, dnextu), Duplicated(nextv, dnextv),
        Duplicated(lastu, dlastu), Duplicated(lastv, dlastv),
        dx, dy, dt, μ, nx, ny
    )
    return nextu, nextv, dlastu, dlastv
end

u,v, du,dv = main()