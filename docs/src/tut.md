# Tutorial

Here we describe how the methods in this package can be used. We illustrate this on the 12,000 cells per well dataset from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040). We only show how we could fit a delayed Fisher-Kolmogorov model. Instructions for fitting, for example, a model with the basis function approach can be found by looking at the corresponding code from our paper as described [here](https://danielvandh.github.io/EquationLearning.jl/dev/paper.html). We start with the following:

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

The field `meshPoints` gives the grid points for the discretised PDE, `LHS` gives the coefficients in the boundary condition $a_0u(a, t) - b_0\partial u(a, t)/\partial x = c_0$, `RHS` gives the coefficients in the boundary condition $a_1u(b, t) + b_1\partial u(b, t)/\partial x = c_1$, `finalTime` gives the time that the solution is solved up to, `δt` gives the vector of points to return the solution at, and `alg` gives the algorithm to use for solving the system of ODEs arising from the discretised PDE. We use the following:

```julia
δt = LinRange(0.0, 48.0 / t_scale, 5)
finalTime = 48.0 / t_scale
N = 500
LHS = [0.0, 1.0, 0.0]
RHS = [0.0, -1.0, 0.0]
alg = Tsit5()
meshPoints = LinRange(75.0 / x_scale, 1875.0 / x_scale, N)
pde_setup = PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)
```

Note that these boundary conditions `LHS` and `RHS` correspond to no flux boundary conditions. For `PDE_Setup` we also provide a constructor with defaults:

- `meshPoints = LinRange(extrema(x)..., 500)`.
- `LHS = [0.0, 1.0, 0.0]`.
- `RHS = [0.0, -1.0, 0.0]`.
- `finalTime = maximum(t)`.
- `δt = finalTime / 4.0`.
- `alg = nothing`.

See the manual for more details.

## Bootstrapping parameters 

The next step is to define the parameters for bootstrapping. The following struct is used for these parameters:

```julia
struct Bootstrap_Setup
    bootₓ::AbstractVector
    bootₜ::AbstractVector
    B::Int
    τ::Tuple{Float64,Float64}
    Optim_Restarts::Int
    constrained::Bool
    obj_scale_GLS::Function
    obj_scale_PDE::Function
    init_weight::Float64
    show_losses::Bool
end
```

The `bootₓ` field gives the spatial grid for bootstrapping, `bootₜ` the spatial grid for bootstrapping, `B` the number of bootstrap iterations, `τ` the two threshold parameters for data thresholding, `Optim_Restarts` the number of the times the optimiser should be restarted, `constrained` an indicator for whether the optimisation problem should be constrained, `obj_scale_GLS` the transformation to apply to the GLS loss function, `obj_scale_PDE` the transformation to apply to the PDE loss function, `init_weight` the weighting to apply in the GLS loss function to the data at $t = 0$, and `show_losses` an indicator for whether the losses should be printed to the REPL at each stage of the optimiser.

For this problem we will use $n = m = 50$ points in space and time for the bootstrapping grid, and $100$ bootstrap iterations with no optimiser restarts. We do not constrain the parameter estimates. To put the loss functions on roughly the same scale we will apply a log transformation to each individual loss function. We weight the data at $t=0$ by a factor of $10$, and we do not show the losses in the REPL. We set this up as follows:

```julia
nₓ = 50
nₜ = 50
bootₓ = LinRange(75.0 / x_scale, 1875.0 / x_scale, nₓ)
bootₜ = LinRange(0.0, 48.0 / t_scale, nₜ)
B = 100
τ = (0.0, 0.0)
Optim_Restarts = 1
constrained = false
obj_scale_GLS = log
obj_scale_PDE = log
init_weight = 10.0
show_losses = false
bootstrap_setup = Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, init_weight, show_losses)
```

This struct also have a constructor, for which the defaults are:

- `bootₓ = LinRange(extrema(x)..., 80)`
- `bootₜ = LinRange(extrema(t)..., 80)`.
- `B = 100`.
- `τ = (0.0, 0.0)`.
- `Optim_Restarts = 5`.
- `constrained = false`.
- `obj_scale_GLS = x -> x`.
- `obj_scale_PDE = x -> x`.
- `init_weight = 10.0`.
- `show_losses = false`.

See the manual for more details.

## Gaussian process parameters 

The Gaussian process parameters are setup in the `GP_Setup` struct, defined as:

```julia
struct GP_Setup
    ℓₓ::Vector{Float64}
    ℓₜ::Vector{Float64}
    σ::Vector{Float64}
    σₙ::Vector{Float64}
    GP_Restarts::Int
    μ::Union{Missing,Vector{Float64}}
    L::Union{Missing,LowerTriangular{Float64}}
    nugget::Float64
    gp::Union{Missing,GPBase}
end
```

The `ℓₓ` field gives a vector defining the interval to sample the spatial length scales between, `ℓₜ` the vector defining the interval to sample the spatial length scales between, `σ` the standard deviation of the noise-free data, `σₙ` the standard deviation of the noise, `GP_Restarts` the number of time to refit the Gaussian process to improve the hyperparameter estimates, `μ` the mean vector of the Gaussian process and derivatives (or missing if it should be computed in `bootstrap_gp` itself), `L` the Cholesky vector of the Gaussian process and derivatives (or missing if it should be computed in `bootstrap_gp` itself), `nugget` the nugget term for regularising the covariance matrix such that it is symmetric positive definite, and `gp` the fitted Gaussian process for just the cell density (or missing if it should be computed in `bootstrap_gp` itself).