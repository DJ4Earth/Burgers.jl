# DiffDistPDE
[![CI](https://github.com/DJ4Earth/Burgers.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/DJ4Earth/Burgers.jl/actions/workflows/ci.yml)

DiffDistPDE is an implementation of a differentiable distributed PDE solver in Julia. It is developed using the latest language features and packages of Julia, which required ample workarounds and is meant as a demonstration of future capabilities for developing differentiable numerical simulation software. It uses
* [MPI.jl](https://github.com/JuliaParallel/MPI.jl) for distributed computing,
* [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) for differentiation,
* and [Checkpointing.jl](https://github.com/Argonne-National-Laboratory/Checkpointing.jl) for reducing the memory footprint of the differentiated code.
