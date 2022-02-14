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
##  - bootstrap_helper: Calls the above three functions.
##  - bootstrap_gp: Does bootstrapping to learn the functional forms.
##
#####################################################################

"""
    preallocate_bootstrap(nₓnₜ, α₀, β₀, γ₀, B)

Creates cache arrays and computes certain parameters 
that are used for the bootstrapping component. See [`bootstrap_helper`](@ref) for details and
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
    praellocate_eqlearn(num_restarts, σ₁, σ₂, σ₃, meshPoints, δt, finalTime, Xₛ, tt, d, r, nₓnₜ, gp)

Creates cache arrays and computes certain parameters that are used for the bootstrapping component. 
See [`bootstrap_helper`](@ref) for details and [`bootstrap_gp`](@ref) for its use. 
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
    bootstrap_helper(x, t, bootₓ, bootₜ, α₀, β₀, γ₀, B, num_restarts, meshPoints, δt, finalTime, gp, lowers, uppers)

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
function bootstrap_helper(x, t, bootₓ, bootₜ, α₀, β₀, γ₀, B, num_restarts, meshPoints, δt, finalTime, gp, lowers, uppers)
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

"""
    bootstrap_helper(x, t, α₀, β₀, γ₀, lowers, uppers, gp_setup::GP_Setup, bootstrap_setup::Bootstrap_Setup, pde_setup::PDE_Setup)

Method for calling [`bootstrap_helper`](@ref) using the setup structs from [`GP_Setup`](@ref), [`Bootstrap_Setup`](@ref), and [`PDE_Setup`](@ref). 
"""
function bootstrap_helper(x, t, α₀, β₀, γ₀, lowers, uppers, gp_setup::GP_Setup, bootstrap_setup::Bootstrap_Setup, pde_setup::PDE_Setup)
    return bootstrap_helper(x, t, bootstrap_setup.bootₓ, bootstrap_setup.bootₜ, α₀, β₀, γ₀,
        bootstrap_setup.B, bootstrap_setup.Optim_Restarts, pde_setup.meshPoints,
        pde_setup.δt, pde_setup.finalTime, gp_setup.gp,
        lowers, uppers)
end

"""
    bootstrap_gp(x::T1, t::T1, u::T2, T::T2, D::T2, D′::T2, R::T2, α₀::T1, β₀::T1, γ₀::T1, lowers::T1, uppers::T1; <keyword arguments>)

Perform bootstrapping on the data `(x, t, u)` to learn the appropriate functional forms of 
`(T, D, R)` with uncertainty. The type signatures `T1` and `T2` refer to `AbstractVector`s and `Function`s, respectively.

# Arguments 
- `x::T1`: The spatial data.
- `t::T1`: The temporal data.
- `u::T1`: The density data.
- `T::T2`: The delay function, given in the form `T(t, α, T_params)`.
- `D::T2`: The diffusion function, given in the form `D(u, β, D_params)`.
- `D′::T2`: The derivative of the diffusion function, given in the form `D′(u, β, D_params)`.
- `R::T2`: The reaction function, given in the form `R(u, γ, R_params)`.
- `α₀::T1`: Initial estimates of the delay parameters. Not actually used for anything other than ensuring the functions are specified correctly.
- `β₀::T1`: Initial estimates of the diffusion parameters. Not actually used for anything other than ensuring the functions are specified correctly. 
- `γ₀::T1`: Initial estimates of the reaction parameters. Not actually used for anything other than ensuring the functions are specified correctly. 
- `lowers::T1`: Lower bounds to use for constructing the Latin hypersquare design, and for the constrained problem if `constrained = true` in `bootstrap_setup`.
- `uppers::T1`: Upper bounds to use for constructing the Latin hypersquare design, and for the constrained problem if `constrained = true` in `bootstrap_setup`.

# Keyword Arguments 
- `gp_setup::GP_Setup = GP_Setup(u)`: Defines the setup for the Gaussian process. See also [`GP_Setup`](@ref).
- `bootstart_setup::Bootstrap_Setup = Bootstrap_Setup(x, t, u)`: Defines some extra keyword arguments for the bootstrapping and optimisation process. See also [`Bootstrap_Setup`](@ref).
- `optim_setup::Optim.Options = Optim_Options()`: Defines some options when using `Optim.optimize`.
- `pde_setup::PDE_Setup = PDE_Setup(x)`: Defines some extra keyword arguments for the PDE solutions. See also [`PDE_Setup`](@ref).
- `D_params = nothing`: Extra known parameters for the diffusion function `D`.
- `R_params = nothing`: Extra known parameters for the reaction function `R`.
- `T_params = nothing`: Extra known parameters for the delay function `T`.
- `PDEkwargs...`: Extra keyword arguments to use inside `DifferentialEquations.solve`.

# Outputs
- `bgp`: A `BootResults` structure. See [`BootResults`](@ref). 
"""
function bootstrap_gp(x::T1, t::T1, u::T2,
    T::T2, D::T2, D′::T2, R::T2, α₀::T1, β₀::T1, γ₀::T1, lowers::T1, uppers::T1;
    gp_setup::GP_Setup = GP_Setup(u),
    bootstrap_setup::Bootstrap_Setup = Bootstrap_Setup(x, t, u),
    optim_setup::Optim.Options = Optim.Options(),
    pde_setup::PDE_Setup = PDE_Setup(x),
    D_params = nothing, R_params = nothing, T_params = nothing, PDEkwargs...) where {T1<:AbstractVector,T2<:Function}
    ## Check provided functions and ODE algorithm are correct
    @assert !(typeof(PDE_Setup.alg) <: Sundials.SundialsODEAlgorithm) "Automatic differentiation is not compatible with Sundials solvers."
    @assert length(x) == length(t) == length(u) "The lengths of the provided data vectors must all be equal."
    try
        D(u[1], β₀, D_params)
    catch
        throw("Either the provided vector of diffusion parameters, β₀ = $β₀, is not of adequate size, or D_params = $D_params has been incorrectly specified.")
    end
    try
        R(u[1], γ₀, R_params)
    catch
        throw("Either the provided vector of reaction parameters, γ₀ = $γ₀, is not of adequate size, or R_params = $R_params has been incorrectly specified.")
    end
    try
        T(t[1], α₀, T_params)
    catch
        throw("Either the provided vector of delay parameters, α₀ = $α₀, is not of adequate size, or T_params = $T_params has been incorrectly specified.")
    end

    ## Compute indices for finding nearest points in the spatial mesh to the actual spatial data x, along with indices for specific values of time.
    time_values = Array{Bool}(undef, length(t), length(PDE_Setup.δt))
    closest_idx = Vector{Vector{Int64}}(undef, length(PDE_Setup.δt))
    iterate_idx = Vector{Vector{Int64}}(undef, length(PDE_Setup.δt))
    for j = 1:length(unique(t))
        @views time_values[:, j] .= t .== PDE_Setup.δt[j]
        @views closest_idx[j] = searchsortednearest.(Ref(PDE_Setup.meshPoints), x[time_values[:, j]]) # Use Ref() so that we broadcast only on x[time_values[:, j]] and not the mesh points
        @views iterate_idx[j] = findall(time_values[:, j])
    end

    ## Fit the GP 
    gp_setup.gp = ismissing(gp_set.gp) ? fit_GP(x, t, u, gp_setup) : gp_setup.gp
    σₙ = exp(gp_setup.gp.logNoise)

    ## Setup the bootstrapping grid, define some parameters, construct cache arrays, etc.
    x_min, x_max, t_min, t_max, x_rng, t_rng, Xₛ, unscaled_t̃, nₓnₜ,
    f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
    tt, d, r,
    delayBases, diffusionBases, reactionBases,
    ℓz, zvals,
    obj_values, stacked_params,
    N, Δx, V,
    Du, D′u, Ru, TuP, DuP, D′uP, RuP, RuN,
    SSEArray, Xₛ₀, IC1,
    initialCondition, MSE = bootstrap_helper(x, t, α₀, β₀, γ₀, lowers, uppers, gp_setup, bootstrap_setup, pde_setup)

    ## Compute the mean vector and Cholesky factor for the joint Gaussian process of the function and its derivatives 
    if ismissing(gp_setup.μ) || ismissing(gp_setup.L)
        gp_setup.μ, gp_setup.L = compute_joint_GP(gp_setup.gp, Xₛ; nugget = gp_setup.nugget)
    end

    ## Now do the equation learning 
    nodes, weights = gausslegendre(length(RuN.du))
    for j = 1:B
        # Draw from N(0, 1) for sampling from the Gaussian process 
        @views randn!(zvals[:, j])

        # Compute the required functions and derivatives 
        @views draw_gp!(ffₜfₓfₓₓ, gp_setup.μ, gp_setup.L, zvals[:, j], ℓz)
        f .= ffₜfₓfₓₓ[f_idx]
        fₜ .= ffₜfₓfₓₓ[fₜ_idx] / t_rng
        fₓ .= ffₜfₓfₓₓ[fₓ_idx] / x_rng
        fₓₓ .= ffₜfₓfₓₓ[fₓₓ_idx] / x_rng^2

        # Threshold the data 
        inIdx = data_thresholder(f, fₜ, τ)

        # Compute the initial condition for the PDE
        @views IC1 .= max.(f[Xₛ₀], 0.0)
        initialCondition .= Dierckx.Spline1D(bootstrap_setup.bootₓ, IC1; k = 1)(pde_setup.meshPoints)

        # Preallocate error vector 
        errs = DiffEqBase.dualcache(zeros(length(inIdx)), trunc(Int64, length(inIdx) / 10)) # We use dualcache since we want to use this within an automatic differentiation computation. 

        ## Parameter estimation 
        @views learn_equations!(x, t, u,
            f, fₜ, fₓ, fₓₓ,
            T, D, D′, R, T_params, D_params, R_params,
            delayBases[:, j], diffusionBases[:, j], reactionBases[:, j], stacked_params,
            lowers, uppers, bootstrap_setup.constrained, obj_values,
            bootstrap_setup.obj_scale_GLS, bootstrap_setup.obj_scale_PDE,
            N, V, Δx, pde_setup.LHS, pde_setup.RHS, initialCondition, 
            pde_setup.finalTime, pde_setup.alg, pde_setup.δt,
            SSEArray,
            Du, Ru, TuP, DuP, RuP, D′uP, RuN,
            inIdx, unscaled_t̃, tt, d, r,
            errs, MSE, optim_setup,
            iterate_idx, closest_idx, nodes, weights, show_losses, σₙ,
            PDEkwargs...)

        print("Bootstrapping: Step $j of $B. Previous objective value: $(minimum(obj_values)).\u001b[1000D")
    end

    Xₛⁿ = deepcopy(Xₛ)
    @muladd @views @. Xₛⁿ[1, :] = Xₛ[1, :] * x_rng + x_min
    @muladd @views @. Xₛⁿ[2, :] = Xₛ[2, :] * t_rng + t_min

    ## Return the results 
    return BootResults(delayBases, diffusionBases, reactionBases, gp, zvals, Xₛ, Xₛⁿ, bootₓ, bootₜ, T, D, D′, R, D_params, R_params, T_params, μ, L, gp_setup, bootstrap_setup, pde_setup)
end