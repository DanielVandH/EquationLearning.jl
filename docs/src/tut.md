# Tutorial

Here we describe how the methods in this package can be used. We illustrate this on the 12,000 cells per well dataset from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040). We only show how we could fit a delayed Fisher-Kolmogorov model. We start with the following:

```julia 
# Packages
using EquationLearning      # Load our actual package 
using DelimitedFiles        # For loading the density data of Jin et al. (2016).
using DataFrames            # For conveniently representing the data
using CairoMakie            # For creating plots
using LaTeXStrings          # For adding LaTeX labels to plots
using Random                # For setting seeds 
using LinearAlgebra         # For setting number of threads to prevent StackOverflowError
using Setfield              # For modifying immutable structs
# Plots and setup
colors = [:black, :blue, :red, :magenta, :green]
LinearAlgebra.BLAS.set_num_threads(1)
# Read in the data 
function prepare_data(filename) # https://discourse.julialang.org/t/failed-to-precompile-csv-due-to-load-error/70146/2
    data, header = readdlm(filename, ',', header=true)
    df = DataFrame(data, vec(header))
    df_new = identity.(df)
    return df_new
end
assay_data = Vector{DataFrame}([])
x_scale = 1000.0 # μm ↦ mm 
t_scale = 24.0   # hr ↦ day 
for i = 1:6
    file_name = string("data/CellDensity_", 10 + 2 * (i - 1), ".csv")
    dat = prepare_data(file_name)
    dat.Position = convert.(Float64, dat.Position)
    dat.Time = convert.(Float64, dat.Time)
    dat.Position ./= x_scale
    dat.Dens1 .*= x_scale^2
    dat.Dens2 .*= x_scale^2
    dat.Dens3 .*= x_scale^2
    dat.AvgDens .*= x_scale^2
    dat.Time ./= t_scale
    push!(assay_data, dat)
end
K = 1.7e-3 * x_scale^2 # Cell carrying capacity as estimated from Jin et al. (2016).
dat = assay_data[2] # The data we will be using in this tutorial
```

## PDE parameters 

The first step is to define the PDE setup. Our function needs a `PDE_Setup` struct from the following function:

```julia
struct PDE_Setup
    meshPoints::AbstractVector
    LHS::Vector{Float64}
    RHS::Vector{Float64}
    finalTime::Float64
    δt::AbstractVector
    alg
end
```

The field `meshPoints` gives the grid points for the discretised PDE, `LHS` gives the coefficients in the boundary condition `$a_0u(a, t) - b_0\partial u(a, t)/\partial x = c_0`, `RHS` gives the coefficients in the boundary condition `a_1u(b, t) + b_1\partial u(b, t)/\partial x = c_1$`, `finalTime` gives the time that the solution is solved up to, `δt` gives the vector of points to return the solution at, and `alg` gives the algorithm to use for solving the system of ODEs arising from the discretised PDE. We use the following:

```julia
δt = LinRange(0.0, 48.0 / t_scale, 5)
finalTime = 48.0 / t_scale
N = 1000
LHS = [0.0, 1.0, 0.0]
RHS = [0.0, -1.0, 0.0]
alg = Tsit5()
meshPoints = LinRange(75.0 / x_scale, 1875.0 / x_scale, 500)
pde_setup = PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)
```

Note that these boundary conditions `LHS` and `RHS` correspond to no flux boundary conditions.