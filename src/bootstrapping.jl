#####################################################################
## Script description: bootstrapping.jl 
##
## This script contains certain functions used for the bootstrap
## equation learning method.
##
## The following functions are defined:
##  - bootstrap_grid: Used for computing the bootstrapping grid.
##  - preallocate_bootstrap: Used for preallocating some important 
##      arrays for bootstrapping.
##  - preallocate_eqlearn: Used for preallocating some arrays that are 
##      used for the equation learning component.
##  - bootstrap_setup: Calls the above three functions.
##
#####################################################################

"""
    preallocate_bootstrap(nₓnₜ, α₀, β₀, γ₀, B)

Creates cache arrays and computes certain parameters 
that are used for the bootstrapping component. See [`bootstrap_setup`](@ref) for details and
[`bootstrap_gp`](@ref) for its use. 
"""
function preallocate_bootstrap(nₓnₜ, α₀, β₀, γ₀, B)
    # Function values
    f = zeros(nₓnₜ)
    fₜ = zeros(nₓnₜ)
    fₓ = zeros(nₓnₜ)
    fₓₓ = zeros(nₓnₜ)
    ffₜfₓfₓₓ = zeros(4nₓnₜ) # For storing [f; fₜ; fₓ; fₓₓ].

    # Indices for extracting function values 
    f_idx = 1:nₓnₜ
    fₜ_idx = (nₓnₜ+1):(2nₓnₜ)
    fₓ_idx = (2nₓnₜ+1):(3nₓnₜ)
    fₓₓ_idx = (3nₓnₜ+1):(4nₓnₜ)

    # Bases 
    tt = length(α₀)
    d = length(β₀)
    r = length(γ₀)
    delayBases = zeros(tt, B) # For storing all the bases (with each column being a set of parameters)
    diffusionBases = zeros(d, B)
    reactionBases = zeros(r, B)

    # Drawing samples
    ℓz = zeros(4nₓnₜ) # Storing the matrix-vector product Lz
    zvals = zeros(4nₓnₜ, B) # Storing the random standard normal variables

    # Return 
    return f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
    tt, d, r,
    delayBases, diffusionBases, reactionBases,
    ℓz, zvals
end

"""
    prellocate_eqlearn(num_restarts, σ₁, σ₂, σ₃, meshPoints, δt, finalTime, Xₛ, tt, d, r, nₓnₜ, gp)

Creates cache arrays and computes certain parameters that are used for the bootstrapping component. 
See [`bootstrap_setup`](@ref) for details and [`bootstrap_gp`](@ref) for its use. 
"""
function preallocate_eqlearn(num_restarts, meshPoints, δt, finalTime, Xₛ, tt, d, r, nₓnₜ, gp, lowers, uppers)
    @assert length(lowers) == length(uppers) == tt + d + r

    # Optimisation
    obj_values = zeros(num_restarts)
    plan, _ = LHCoptim(num_restarts, tt + d + r, 1000) # Construct the initial design
    stacked_params = Matrix(scaleLHC(plan, [(lowers[i] + 1e-5, uppers[i] - 1e-5) for i in 1:(tt+d+r)])') # Scale into the correct domain. Note that we shift values by 1e-5 so that the points don't lie on the boundary

    # PDE
    N = length(meshPoints)
    Δx = diff(meshPoints)
    V = @views 1 / 2 * [Δx[1]; Δx[1:(N-2)] + Δx[2:(N-1)]; Δx[N-1]]
    Du = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10)) # We use dualcache so that we can easily integrate automatic differentiation into our ODE solver. The last argument is the chunk size, see the ForwardDiff docs for details. The /10 is just some heuristic I developed based on warnings given by PreallocationTools.
    D′u = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
    Ru = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
    TuP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
    DuP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
    D′uP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
    RuP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
    RuN = DiffEqBase.dualcache(zeros(5), 6)
    SSEArray = DiffEqBase.dualcache((zeros(length(meshPoints), δt isa Number ? length(0:δt:finalTime) : length(δt))), trunc(Int64, N / 10))
    Xₛ₀ = Xₛ[2, :] .== 0.0
    IC1 = zeros(count(Xₛ₀))
    initialCondition = zeros(N)
    MSE = DiffEqBase.dualcache(zeros(size(gp.x, 2)), trunc(Int64, size(gp.x, 2) / 10))

    # Return 
    return obj_values, stacked_params,
    N, Δx, V,
    Du, D′u, Ru, TuP, DuP, D′uP, RuP, RuN,
    SSEArray, Xₛ₀, IC1, initialCondition, MSE
end

"""
    bootstrap_setup(x, t, bootₓ, bootₜ, α₀, β₀, γ₀, B, num_restarts, meshPoints, δt, finalTime, gp, lowers, uppers)

Computes all the required cache arrays and certain parameter values for the bootstrapping process. 
The function simply calls [`bootstrap_grid`](@ref), [`preallocate_bootstrap`](@ref), and [`preallocate_eqlearn`](@ref). See also 
[`bootstrap_gp`](@ref).

# Arguments 
- `x`: The original spatial data. 
- `t`: The original temporal data.
- `bootₓ`: The spatial bootstrapping grid.
- `bootₜ`: The temporal bootstrapping grid.
- `α₀`: Initial values for the delay coefficients. (Not actually used anywhere other than for computing the number of delay parameters and checking arguments.)
- `β₀`: Initial values for the diffusion coefficients. (Not actually used anywhere other than for computing the number of diffusion parameters and checking arguments.)
- `γ₀`: Initial values for the reaction coefficients. (Not actually used anywhere other than for computing the number of reaction parameters and checking arguments.)
- `B`: Number of bootstrap iterations being performed.
- `num_restarts`: Number of times to restart the optimisation problem when solving the nonlinear least squares problems. See [`learn_equations!`](@ref).
- `meshPoints`: The spatial mesh used for solving the PDEs.
- `δt`: A number or a vector specifying the spacing between returned times for the solutions to the PDEs or specific times, respectively.
- `finalTime`: The final time to give the solution to the PDEs at.
- `gp`: The fitted Gaussian process. See [`fit_GP`](@ref).
- `lowers`: Lower bounds for the delay, diffusion, and reaction parameters (in that order) for constructing the Latin hypercube design (only for the grid, these do not constrain the parameter values).
- `uppers`: Upper bounds for the delay, diffusion, and reaction parameters (in that order) for constructing the Latin hypercube design (only for the grid, these do not constrain the parameter values).

# Outputs
The outputs are broken into categories.

*Bootstrapping computation*:
- `x_min`: The minimum `x` value.
- `x_max`: The maximum `x` value.
- `t_min`: The minimum `t` value.
- `t_max`: The maximum `t` value.
- `x_rng`: The range of the `x` values, `x_max - x_min`.
- `t_rng`: The range of the `t` values, `t_max - t_min`.
- `Xₛ`: The test matrix for the bootstrapping grid data.
- `unscaled_t̃`: The unscaled `t` values for the bootstrapping grid. Used for computing the delay function in the loss function.
- `nₓnₜ`: The number of test data points.
*Functions*:
- `f`: Cache array for `f(x, t)`.
- `fₜ`: Cache array for `fₜ(x, t)`.
- `fₓ`: Cache array for `fₓ(x, t)`.
- `fₓₓ`: Cache array for `fₓₓ(x, t)`.
- `ffₜfₓfₓₓ`: Cache array for the stacked vector `[f; fₜ; fₓ; fₓₓ]`.
- `f_idx`: Indices for extracting `f(x, t)` from `ffₜfₓfₓₓ` from the random samples. 
- `fₜ_idx`: Indices for extracting `fₜ(x, t)` from `ffₜfₓfₓₓ` from the random samples.
- `fₓ_idx`: Indices for extracting `fₓ(x, t)` from `ffₜfₓfₓₓ` from the random samples.
- `fₓₓ_idx`: Indices for extracting `fₓₓ(x, t)` from `ffₜfₓfₓₓ` from the random samples.
*Bases*:
- `tt`: Number of delay parameters.
- `d`: Number of diffusion parameters.
- `r`: Number of reaction parameters.
- `delayBases`: Matrix for storing the computed delay coefficients. Each column represents a set of parameters.
- `diffusionBases`: Matrix for storing the computed diffusion coefficients. Each column represents a set of parameters.
- `reactionBases`: Matrix for storing the computed reaction coefficients. Each column represents a set of parameters.
*Samples*:
- `ℓz`: Cache array for storing the result of the matrix-vector product `Lz`, where `L` is the Cholesky factor and `z` is a random sample from `N(0, I)`.
- `zvals`: Matrix for storing the drawn `z` values from `N(0, I)`.
Optimisation:
- `obj_values`: Cache array for storing the objective function values at each optimisation restart.
- `stacked_params`: Matrix which stores parameter values at each optimisation restart. The columns take the form `[α; β; γ]`.
*PDE geometry*:
- `N`: The length of `meshPoints`.
- `Δx`: The spacing between each point in `meshPoints`.
- `V`: The volume of each cell in the spatial mesh.
*PDE computation*:
- `Du`: Cache array for computing `D(u)`.
- `D′u`: Cache array for computing `D′(u)`.
- `Ru`: Cache array for computing `R(u)`.
- `TuP`: Cache array for storing the values of the delay function at the unscaled times (for the `"PDE"` loss function).
- `DuP`: Cache array for storing the values of the diffusion function at the estimated density values (for the `"PDE"` loss function).
- `D′uP`: Cache array for storing the values of the derivative of the diffusion function at the estimated density values (for the `"PDE"` loss function).
- `RuP`: Cache array for storing the values of the reaction function at the estimated density values (for the `"PDE"` loss function).
- `RuN`: For storing values of the reaction function at Gauss-Legendre quadrature nodes.
*PDE loss function*:
- `SSEArray`: Cache array for storing the solutions to the PDEs.
- `Xₛ₀`: Logical array used for accessing the values in `Xₛ` corresponding to the initial condition.
- `IC1`: Cache array for storing the initial spline over the initial data.
- `initialCondition`: Cache array for storing the initial condition over `meshPoints`.
- `MSE`: Cache array for storing the individual squared errors.
"""
function bootstrap_setup(x, t, bootₓ, bootₜ, α₀, β₀, γ₀, B, num_restarts, meshPoints, δt, finalTime, gp, lowers, uppers)
    x_min, x_max, t_min, t_max,
    x_rng, t_rng, Xₛ, unscaled_t̃, nₓnₜ = bootstrap_grid(x, t, bootₓ, bootₜ)
    f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
    tt, d, r,
    delayBases, diffusionBases, reactionBases,
    ℓz, zvals = preallocate_bootstrap(nₓnₜ, α₀, β₀, γ₀, B)
    obj_values, stacked_params,
    N, Δx, V,
    Du, D′u, Ru, TuP, DuP, D′uP, RuP, RuN,
    SSEArray, Xₛ₀, IC1,
    initialCondition, MSE = preallocate_eqlearn(num_restarts, meshPoints, δt, finalTime, Xₛ, tt, d, r, nₓnₜ, gp, lowers, uppers)
    return x_min, x_max, t_min, t_max, x_rng, t_rng, Xₛ, unscaled_t̃, nₓnₜ,
    f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
    tt, d, r,
    delayBases, diffusionBases, reactionBases,
    ℓz, zvals,
    obj_values, stacked_params,
    N, Δx, V,
    Du, D′u, Ru, TuP, DuP, D′uP, RuP, RuN,
    SSEArray, Xₛ₀, IC1,
    initialCondition, MSE
end