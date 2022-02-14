#####################################################################
## Script description: structs.jl
##
## This script contains some structure definitions along with some 
## constructors. Some of these functions are used for setting keyword
## arguments conveniently in a way that doesn't clutter function 
## definitions.

## The following structs are define:
##  - BootResults: A struct for storing bootstrapping results.
##  - GP_Setup: A struct that defines the setup for a Gaussian process. 
##      A constructor is also defined.
##  - Bootstrap_Setup: A struct for defining the setup for bootstrapping.
##      A constructor is also defined.
##  - PDE_Setuo: A struct for defining the configuration used for PDEs.
##      A constructor is also defined.
#####################################################################

"""
    mutable struct GP_Setup{T}

Setup for the Gaussian processes. See also [`fit_GP`](@ref) and [`bootstrap_gp`](@ref).

# Fields 
- `ℓₓ::Vector{T}`: A 2-vector giving the lower and upper bounds for the initial estimates of `ℓₓ` (defined on a log scale).
- `ℓₜ::Vector{T}`: A 2-vector giving the lower and upper bounds for the initial estimates of `ℓₜ` (defined on a log scale).
- `σ::Vector{T}`: A 2-vector giving the lower and upper bounds for the initial estimates of `σ` (defined on a log scale).
- `σₙ::Vector{T}`: A 2-vector giving the lower and upper bounds for the initial estimates of `σₙ` (defined on a log scale).
- `GP_Restarts::Int`: Number of times to restart the optimiser. See [`opt_restart!`](@ref).
- `μ::Union{Missing, Vector{T}}`: Either `Nothing` or a stored mean vector for the Gaussian process. 
- `L::Union{Missing, LowerTriangular{T}}`: Either `Nothing` or a stored Cholesky factor for the Gaussian process.
- `nugget::T`: A term to add to the correlation matrix to force the covariance matrix to be symmetric positive definite.
- `gp::Union{Missing, GPBase}`: Either `Nothing` or a stored Gaussian process. See also [`fit_GP`](@ref).
"""
mutable struct GP_Setup{T} # We use mutable so that we can update it with the found μ, L, gp if we want
    ℓₓ::Vector{T}
    ℓₜ::Vector{T}
    σ::Vector{T}
    σₙ::Vector{T}
    GP_Restarts::Int
    μ::Union{Missing,Vector{T}}
    L::Union{Missing,LowerTriangular{T}}
    nugget::T
    gp::Union{Missing,GPBase}
end

"""
    GP_Setup(u; <keyword arguments>)

A constructor for [`GP_Setup`](@ref) for some density data `u`. 
    
# Keyword Arguments
- `ℓₓ = log.([1e-4, 1.0])`.
- `ℓₜ = log.([1e-4, 1.0])`.
- `σ = log.([1e-1, 2std(u)])`.
- `σₙ = log.([1e-5, 2std(u)])`.
- `GP_Restarts = 50`.
- `μ = missing`.
- `L = missing`.
- `nugget = 1e-4`.
- `gp = missing`.
"""
function GP_Setup(u;
    ℓₓ = log.([1e-4, 1.0]),
    ℓₜ = log.([1e-4, 1.0]),
    σ = log.([1e-1, 2std(u)]),
    σₙ = log.([1e-5, 2std(u)]),
    GP_Restarts = 50,
    μ = missing,
    L = missing,
    nugget = 1e-4,
    gp = missing)
    return GP_Setup(promote(ℓₓ, ℓₜ, σ, σₙ)..., Int(GP_Restarts), μ, L, convert(eltype(ℓₓ), nugget), gp)
end

"""
    struct Bootstrap_Setup

A struct defining some arguments for [`bootstrap_gp`](@ref).

# Fields
- `bootₓ::AbstractVector`: The spatial grid for bootstrapping.
- `bootₜ::AbstractVector`: The temporal grid for bootstrapping.
- `B::Int`: Number of bootstrap samples.
- `τ::Tuple{Float64, Float64}`: A tuple of the form `(τ₁, τ₂)` which gives the tolerance `τ₁` for thresholding `f` and `τ₂` for thresholding `fₜ`. See also [`data_thresholder`](@ref).
- `Optim_Restarts::Int`: Number of times to restart the optimiser for the nonlinear least squares problem. See also [`learn_equations!`](@ref).
"""
struct Bootstrap_Setup
    bootₓ::AbstractVector
    bootₜ::AbstractVector
    B::Int
    τ::Tuple{Float64,Float64}
    Optim_Restarts::Int
    constrained::Bool
    obj_scale_GLS::Float64
    obj_scale_PDE::Float64
    show_losses::Bool
end

"""
    Bootstrap_Setup(x::AbstractVector, t::AbstractVector; <keyword arguments>)

A constructor for [`Bootstrap_Setup`](@ref) for some spatial data `x` and temporal data `t`.

# Keyword Arguments
- `bootₓ = LinRange(extrema(x)..., 80)`
- `bootₜ = LinRange(extrema(t)..., 80)`.
- `B = 100`.
- `τ = (0.0, 0.0)`.
- `Optim_Restarts = 5`.
- `constrained = false`.
- `obj_scale_GLS = 1.0`.
- `obj_scale_PDE = 1.0`.
- `show_losses = false`.
"""
function Bootstrap_Setup(x::AbstractVector, t::AbstractVector;
    bootₓ = LinRange(extrema(x)..., 80),
    bootₜ = LinRange(extrema(t)..., 80),
    B = 100,
    τ = (0.0, 0.0),
    Optim_Restarts = 5,
    constrained = false,
    obj_scale_GLS = 1.0,
    obj_scale_PDE = 1.0,
    show_losses = false)
    @assert length(x) == length(t) "The spatial data x and length data t must be vectors of equal length."
    @assert B > 0 "The number of bootstrap samples must be a positive integer."
    @assert all(0 .≤ τ .≤ 1.0) "The threshold tolerances in τ must be between 0.0 and 1.0."
    @assert Optim_Restarts > 0 "The number of restarts mus tbe a positive integer."
    @assert obj_scale_GLS > 0.0 "The scale for the GLS loss must be positive."
    @assert obj_scale_PDE > 0.0 "The scale for the PDE loss msut be positive."
    return Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, show_losses)
end

"""
    struct PDE_Setup 

A struct defining some arguments for the PDEs in [`bootstrap_gp`](@ref) and [`boot_pde_solve`](ref).

# Fields 
- `meshPoints::AbstractVector`: The spatial mesh to use for solving the PDEs involved when computing the `"GLS"` loss function, or for the PDEs in general.
- `LHS::Vector{Float64}`: Vector defining the left-hand boundary conditions for the PDE. See also the definitions of `(a₀, b₀, c₀)` in [`sysdegeneral!`](@ref).
- `RHS::Vector{Float64}`: Vector defining the right-hand boundary conditions for the PDE. See also the definitions of `(a₁, b₁, c₁)` in [`sysdegeneral!`](@ref).
- `finalTime::Float64`: The final time to solve the PDE up to.
- `δt::Union{AbstractVector, Float64}`: A number or a vector specifying the spacing between returned times for the solutions to the PDEs or specific times, respectively.
- `alg`: Algorithm to use for solving the PDEs. If you want to let `DifferentialEquations.jl` select the algorithm automatically, specify `alg = nothing`. If automatic differentiation is being used in the ODE algorithm, then no `Sundials` algorithms can be used.
"""
struct PDE_Setup
    meshPoints::AbstractVector
    LHS::Vector{Float64}
    RHS::Vector{Float64}
    finalTime::Float64
    δt::AbstractVector
    alg
end

"""
    PDE_Setup(x, t; <keyword arguments>)

A constructor for [`PDE_Setup`](@ref) for some spatial data `x` and temporal data `t`.

# Keyword Arguments 
- `meshPoints = LinRange(extrema(x)..., 500)`.
- `LHS = [0.0, 1.0, 0.0]`.
- `RHS = [0.0, -1.0, 0.0]`.
- `finalTime = maximum(t)`.
- `δt = finalTime / 4.0`.
- `alg = nothing`.
"""
function PDE_Setup(x, t;
    meshPoints = LinRange(extrema(x)..., 500),
    LHS = [0.0, 1.0, 0.0],
    RHS = [0.0, -1.0, 0.0],
    finalTime = maximum(t),
    δt = finalTime / 4.0,
    alg = nothing)
    @assert length(LHS) == length(RHS) == 3 "The provided boundary condition vectors LHS and RHS must be of length 3."
    if δt isa Number
        δt = 0:δt:finalTime
        if length(δt) ≠ length(unique(t)) 
            error("Length of δt must be the same as the number of unique time points in t.")
        end
    end
    return PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)
end

"""
    struct BootResults

Structure for storing bootstrapping results. See [`bootstrap_gp`](@ref).

# Fields 
- `delayBases`: The estimated delay parameters. Each column corresponds to a single bootstrap iteration.
- `diffusionBases`: The estimated diffusion parameters. Each column corresponds to a single bootstrap iteration.
- `reactionBases`: The estimated reaction parameters. Each column corresponds to a single bootstrap iteration.
- `gp`: The fitted Gaussian process. See [`fit_gp`](@ref).
- `zvals`: The simulated normal variables from `N(0, I)` used for drawing from the Gaussian process `gp`. See [`draw_gp!`](@ref).
- `Xₛ`: The test matrix for the bootstrapping grid data, given in the scale [0, 1].
- `Xₛⁿ`: The unscaled form of `Xₛ`.
- `bootₓ`: The spatial bootstrapping grid.
- `bootₜ`: The temporal bootstrapping grid.
- `T`: The delay function, given in the form `T(t, α, T_params)`.
- `D`: The diffusion function, given in the form `D(u, β, D_params)`.
- `D′`: The derivative of the diffusion function, given in the form `D′(u, β, D_params)`.
- `R`: The reaction function, given in the form `R(u, γ, R_params)`.
- `D_params`: Parameters for the diffusion function.
- `R_params`: Parameters for the reaction function. 
- `T_params`: Parameters for the delay function.
- `μ`: The mean vector for the joint Gaussian process. See also [`compute_joint_GP`](@ref).
- `L`: The Cholesky factor for the joint Gaussian process. See also [`compute_joint_GP`](@ref).
"""
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
    D_params
    R_params
    T_params
    μ::Vector{Float64}
    L::LowerTriangular{Float64}
    gp_setup::GP_Setup
    bootstrap_setup::Bootstrap_Setup 
    pde_setup::PDE_Setup
end