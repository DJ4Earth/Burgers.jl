using Adapt
using Enzyme
using KernelAbstractions
using CUDA

const KA = KernelAbstractions

struct Model{MT}
    lastu::MT
    nextu::MT
    nx::Int64
    ny::Int64
    backend::KA.Backend
end

function Adapt.adapt(backend::KA.Backend, model::Model)
    Model(
        adapt(backend, model.lastu),
        adapt(backend, model.nextu),
        model.nx,
        model.ny,
        backend,
    )
end

function Model(lastu, nextu, nx, ny, backend=CPU())
    _lastu = adapt(backend, lastu)
    _nextu = adapt(backend, nextu)
    return Model(_lastu, _nextu, nx, ny)
end

@kernel function stencil_kernel!(nextu, lastu)
    i,j = @index(Global, NTuple)
    nextu[i,j] = 2*lastu[i,j]
end

function stencil!(model)
    stencil_kernel!(model.backend)(
        model.nextu, model.lastu;
        ndrange = (model.nx, model.ny)
    )
    KA.synchronize(model.backend)
    return nothing
end

function main()
    nx = 10
    ny = 10
    lastu = ones(nx, ny)
    nextu = copy(lastu)

    model_cpu = Model(lastu, nextu, nx, ny)
    stencil!(model_cpu)

    model_gpu = adapt(CUDABackend(), model_cpu)
    stencil!(model_gpu)
    # dlastu = copy(lastu)
    # dnextu = copy(nextu)

    # Enzyme.autodiff(
    #     ReverseWithPrimal, stencil!,
    #     Duplicated(nextu, dnextu),
    #     Duplicated(lastu, dlastu),
    #     @Const(nx), @Const(ny),
    #     @Const(backend),
    # )
    return model_cpu.nextu, model_gpu.nextu#, dlastu
end

u_cpu, u_gpu = main()
@show typeof(u_cpu)
@show typeof(u_gpu)
