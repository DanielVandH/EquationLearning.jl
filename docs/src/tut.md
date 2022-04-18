# Tutorial

Here we describe how the methods in this package can be used. We illustrate this on the 12,000 cells per well dataset from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040). We only show how we could fit a delayed Fisher-Kolmogorov model. Instructions for fitting, for example, a model with the basis function approach can be found by looking at the corresponding code from our paper as described [here](https://danielvandh.github.io/EquationLearning.jl/dev/paper.html). We start with the following:

```julia 
# Packages
using EquationLearning      # Load our actual package 
using DelimitedFiles        # For loading the density data of Jin et al. (2016).
using DataFrames            # For conveniently representing the data
using LinearAlgebra         # For setting number of threads to prevent StackOverflowError
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
    file_name = string("data/CellDensity_", 10 + 2 * (i - 1), ".csv") # This assumes that you are in the VandenHeuvel2022_Paper code folder of our repository. If the data is hosted somewhere else, simply change this line to locate the correct directory.
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

For this data we also need to extract $x$, $t$, $u$, and the values to use for the PDEs:

```julia
x = repeat(dat.Position, outer=3)
t = repeat(dat.Time, outer=3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens
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

Since we scale the data to be between $0$ and $1$ when fitting the Gaussian process, we pick our length scales to be between $0$ and $1$; these length scales must be defined on a log scale due to how they are defined in [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl). The values for the standard deviations are less clear and so we will choose it to be small, and also base it on the standard deviation of the observed data. A reasonably small nugget value needs to be used. Moreover, we can use many optimiser restarts since the Gaussian processes are not too expensive to fit. This leads to the following parameters:

```julia
ℓₓ = log.([1e-6, 1.0])
ℓₜ = log.([1e-6, 1.0])
nugget = 1e-5
GP_Restarts = 250
σ = log.([1e-6, 7std(u)])
σₙ = log.([1e-6, 7std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)
```

This struct `GP_Setup` also has a constructor with the following defaults:

- `ℓₓ = log.([1e-6, 1.0])`.
- `ℓₜ = log.([1e-6, 1.0])`.
- `σ = log.([1e-6, 7std(u)])`.
- `σₙ = log.([1e-6, 7std(u)])`.
- `GP_Restarts = 50`.
- `μ = missing`.
- `L = missing`.
- `nugget = 1e-4`.
- `gp = missing`.

See the manual for more details.

## Defining the functions 

Now we want to define the functions and the corresponding parameter scales. Remember that the model we want to fit takes the form

```math 
\frac{\partial u}{\partial t} = T(t; \boldsymbol{\alpha})\left[\frac{\partial}{\partial x}\left(D(u; \boldsymbol{\beta})\frac{\partial u}{\partial x}\right) + R(u; \boldsymbol{\gamma})\right],
```

where 

```math 
T(t; \boldsymbol{\alpha}) = \frac{1}{1+\exp(-\alpha_1-\alpha_2t)}, \quad D(u; \boldsymbol{\alpha}) = \beta_1, \quad R(u; \boldsymbol{\gamma}) = \gamma_1u\left(1-\frac{u}{K}\right).
```

The function `bootstrap_gp` assumes that $T$, $D$, and $R$ are given as functions of $(t, \boldsymbol{\alpha}, \boldsymbol{p})$, $(u, \boldsymbol{\beta}, \boldsymbol{p})$, and $(u, \boldsymbol{\gamma}, \boldsymbol{p})$, respectively. This vector $\boldsymbol{p}$ gives a vector of known parameters for each of the functions, and is used to allow for type stability in the functions. In particular, it would be wrong to define the reaction function as

```julia
R = (u, γ, p) -> γ[1] * u * (1.0 - u / K)
```

since `K` is in the scope of the main Julia REPL rather than the function itself, potentially leading to problems with type stability and making the function significantly slower. With this in mind, we define our functions as follows:

```julia
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
```

The derivative for the reaction term is not currently used, but in the future it may be used and we thus include it as a necessary term in `bootstrap_gp`, hence its presence here. In these functions we give each parameter a corresponding value in the vector $\boldsymbol{p}$, which we can use to put the parameters on the same scale. To now think about defining $\boldsymbol{p}$, we could set the parameter scales to be $1$ and fit a small number of models, as described in great detail in the first simulation study of our paper. We will instead provide a way that should typically good enough (that is only specific to this dataset since there already exist papers that study this data). In Table 1 of [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040), the values for $\beta_1$ and $\gamma_1$ are given as $250 \pm 140$ and $0.044 \pm 0.002$, respectively, in units of $\mu\textrm{m}^2\textrm{h}^{-1}$ and $\textrm{h}^{-1}$, respectively. The delay term should not, affect these estimates so significantly, although we do not expect to get these exact same parameter values, so we should not scale the parameters with exactly these values. Similarly, although they consider a different form of the model, [Lagergren et al. (2020)](https://doi.org/10.1371/journal.pcbi.1008462) present the same delay model for this dataset, finding $\alpha_1 = -3.3013$ and $\alpha_2 = 0.2293$ ($\textrm{hr}^{-1}$). Based on these estimates, and with some other configuring with these estimates (after running the model for a small number of models to further improve these estimates in a significantly faster time than if we simply started with unit parameter scales), we define the following parameters:

```julia
T_params = [-1.6, 0.2 * t_scale]
D_params = [160.0 * t_scale / x_scale^2]
R_params = [K, 0.057 * t_scale]
```

Note that we multiply by `t_scale` and `x_scale` so that we shift the parameters into the correct units.

The remaining required arguments in `bootstrap_gp` are the bounds on parameters, and placeholder vectors for the number of parameters to estimate for each mechanism. For these latter arguments, we simply define:

```julia
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
```

The values do not matter, they just have to have the same values. For the parameter bounds, these do not matter since we will not be doing any optimiser restarts. (These bounds are not constraints on the parameters since we have `constrained = false`; in this case they simply define the hypercube for the Latin hypercube sampler to use for sampling parameter estimates.) We still have to provide both in this case. When we use a single optimiser restart, the initial value used in the optimiser is the middle of the given parameter bounds. Since we scale the parameters to all (hopefully) be $\mathcal O(1)$, we start each parameter at $1$:

```julia
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
```

## Fitting the model

We now fit the model. (Since we are not going to be performing any model comparisons in this tutorial, we do not provide a `zvals` argument; see how we use this argument for model comparison in [VandenHeuvel2022_PaperCode/paper_code.jl](https://github.com/DanielVandH/EquationLearning.jl/blob/5466b87ae7ed3d3d171123ddf3d595d881538490/VandenHeuvel2022_PaperCode/paper_code.jl) by first defining a `zvals` vector and then using `zvals = zvals` in `bootstrap_gp`. These `zvals` are also provided in the struct for the final results, so we could instead not provide them and simply reuse the `zvals` from the first model.) We call the function as:

```julia
optim_setup = Optim.Options(iterations=10, f_reltol=1e-4, x_reltol=1e-4, g_reltol=1e-4, outer_f_reltol=1e-4, outer_x_reltol=1e-4, outer_g_reltol=1e-4)
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
```

We also include some optimisation options in this call. The `verbose=false` argument is used to prevent any spam from the differential equations solver in the REPL in case the parameters enter a region where all the solutions become unstable (which it will eventually exit out of and give reasonable parameter estimates, but this issue can occasionally happen for certain models). Other arguments for the differential equations solver, i.e. for the `solve` function from [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/), can be similarly provided by keyword.

This result `bgp` is a `BootResults` struct, defined as follows:

```julia
struct BootResults
    delayBases::Array{Float64}
    diffusionBases::Array{Float64}
    reactionBases::Array{Float64}
    gp::GPBase
    zvals::Array{Float64}
    Xₛ::Array{Float64}
    Xₛⁿ::Array{Float64}
    bootₓ::Vector{Float64}
    bootₜ::Vector{Float64}
    T::Function
    D::Function
    D′::Function
    R::Function
    R′::Function
    D_params
    R_params
    T_params
    μ::Vector{Float64}
    L::LowerTriangular{Float64}
    gp_setup::GP_Setup
    bootstrap_setup::Bootstrap_Setup
    pde_setup::PDE_Setup
end
```

The meaning for each field is self-explanatory. We also define a similar struct for computing the PDE solutions, plots, and AICs:

```julia
struct AllResults
    pde_solutions::Array{Float64}
    AIC::Vector{Float64}
    bgp::Union{BootResults,BasisBootResults}
    delayCIs
    diffusionCIs
    reactionCIs
    delay_density::Figure
    diffusion_density::Figure
    reaction_density::Figure
    delay_curve::Figure
    diffusion_curve::Figure
    reaction_curve::Figure
    pde_plot::Figure
    pde_error::Vector{Float64}
end
```

We would typically use the constructor `AllResults(x_pde, t_pde, u_pde, bgp)` for constructing this struct. Since we have scaled the data, we also want to make use of additional keyword arguments to put the data back on the original scale. We thus create the `AllResults` struct for this data as:

```julia
delay_scales = [T_params[1], T_params[2] / t_scale]
diffusion_scales = D_params[1] * x_scale^2 / t_scale
reaction_scales = R_params[2] / t_scale
res = AllResults(x_pde, t_pde, u_bgp, bgp; delay_scales, diffusion_scales, reaction_scales, x_scale, t_scale, correct = true)
```

The `correct = true` keyword argument is used so that a small sample size correction is used for the computed AICs.

## Looking at the results

To finish the tutorial, we discuss how we could look at the results. The plots are all stored in this `res` variable we computed above. We do not show the plots here, but we would just look through the plots by writing in the REPL (the results may vary if you are running code alongside this tutorial due to the random number generation):

```julia-repl
julia> res

Bootstrapping results

Number of bootstrap samples: 100
PDE Error: (10.1, 14.5)
AIC: (1.56e+03, 1.68e+03)

α[1]: (-3.09, -0.854)
α[2]: (0.133, 0.307)
β[1]: (117, 222)
γ[1]: (0.0548, 0.0615)


julia> res.pde_error
2-element Vector{Float64}:
 10.104350220097093
 14.546789846145

julia> res.delay_density

julia> res.delay_curve

julia> res.diffusion_density

julia> res.diffusion_curve

julia> res.reaction_density

julia> res.reaction_curve
```

These plots may not always be what is desired by the user. Customising these plots based on the `bgp` results may take some work, and ways that we could for example plot all these plots in the same figure (as we do in the paper), are given in the plotting functions in [VandenHeuvel2022_PaperCode/paper_code.jl](https://github.com/DanielVandH/EquationLearning.jl/blob/5466b87ae7ed3d3d171123ddf3d595d881538490/VandenHeuvel2022_PaperCode/paper_code.jl).
