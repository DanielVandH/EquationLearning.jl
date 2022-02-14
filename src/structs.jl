"""
    struct BootResults

Structure for storing bootstrapping results. See [`bootstrap_gp`](@ref).

# Fields 
- `delayBases`: The estimated delay parameters. Each column corresponds to a single bootstrap iteration.
- `diffusionBases`: The estimated diffusion parameters. Each column corresponds to a single bootstrap iteration.
- `reactionBases`: The estimated reaction parameters. Each column corresponds to a single bootstrap iteration.
- `gp`: The fitted Gaussian process. See [`fit_gp`](@re).
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
end