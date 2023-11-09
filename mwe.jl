using Adapt
using Enzyme
using KernelAbstractions
using CUDA

struct Model{T,MT} where {T,MT}
    lastu::MT{T}
    nextu::MT{T}
    nx::Int64
    ny::Int64
    device::KA.Device
end

function Model(lastu, nextu, nx, ny, device=CPU())
    _lastu = adapt(device, lastu)
    _nextu = adapt(device, nextu)
    return Model(_lastu, _nextu, nx, ny)
end

@kernel function stencil_kernel!(nextu, lastu)
    i,j = @index(Global, NTuple)
    nextu[i,j] = 2*lastu[i,j]
end

function stencil!(model)
    adapt(device, nextu)
    adapt(device, lastu)
    stencil_kernel!(model.device)(
        model.nextu, model.lastu;
        ndrange = (model.nx, model.ny)
    )
    synchronize(device)
    return nothing
end

function main()
    nx = 10
    ny = 10
    lastu = ones(nx, ny)
    nextu = copy(lastu)
    model_cpu = Model(lastu, nextu, nx, ny)
    stencil!(model_cpu)

    # dlastu = copy(lastu)
    # dnextu = copy(nextu)

    # Enzyme.autodiff(
    #     ReverseWithPrimal, stencil!,
    #     Duplicated(nextu, dnextu),
    #     Duplicated(lastu, dlastu),
    #     @Const(nx), @Const(ny),
    #     @Const(device),
    # )
    return model.nextu#, dlastu
end

u = main()
