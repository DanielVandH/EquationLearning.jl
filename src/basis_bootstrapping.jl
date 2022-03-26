"""
    evaluate_basis(coefficients, basis, point)

Evaluates the function `func(u) = ∑_(i=1)^n coefficients[i]basis[i](u)` at the point u = `point`. 
The function uses `@inline` to suggest that the compiler could inline this in the LLVM.
"""
@inline function evaluate_basis(coefficients, basis, point, params)
    return sum(@inbounds(coefficients[k] * basis[k](point, params)) for k in eachindex(coefficients)) # dot product
end

"""
    evaluate_basis!(val, coefficients, basis, point, A)

Evaluates the function `func(u) = ∑_(i=1)^n coefficients[i]basis[i](u)` at the point u = `point` and puts it into `val`.
This function differs from `evaluateBasis` as it works directly on vectors and uses matrix multiplication rather than dot products (although they are equivalent).
The function uses `@inline` to suggest that the compiler could inline this in the LLVM.
"""
@inline function evaluate_basis!(val, coefficients, basis, point, params, A)
    ## Construct the matrix: 
    @inbounds @views for (j, f) in enumerate(basis)
        A[:, j] .= f.(point, Ref(params))
    end
    mul!(val, A, coefficients)
    return nothing
end

"""
    basis_bootstrap_helper(x, t, bootₓ, bootₜ, d, r, B)

Computes all the required cache arrays and certain parameter values for the basis bootstrapping process. 

# Arguments 
- `x`: The original spatial data. 
- `t`: The original temporal data.
- `bootₓ`: The spatial bootstrapping grid.
- `bootₜ`: The temporal bootstrapping grid.
- `d`: Number of diffusion parameters. 
- `r`: Number of reaction parameters.
- `B`: Number of bootstrap iterations being performed.

# Outputs
The outputs are broken into categories.

*Bootstrapping computation*:
- `x_min`: The minimum `x` value.
- `t_min`: The minimum `t` value.
- `x_rng`: The range of the `x` values, `x_max - x_min`.
- `t_rng`: The range of the `t` values, `t_max - t_min`.
- `Xₛ`: The test matrix for the bootstrapping grid data.
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
- `diffusionBases`: Matrix for storing the computed diffusion coefficients. Each column represents a set of parameters.
- `reactionBases`: Matrix for storing the computed reaction coefficients. Each column represents a set of parameters.
*Samples*:
- `ℓz`: Cache array for storing the result of the matrix-vector product `Lz`, where `L` is the Cholesky factor and `z` is a random sample from `N(0, I)`.
- `zvals`: Matrix for storing the drawn `z` values from `N(0, I)`.
*Other caches*
- `Du`: Cache array for computing `D(u)`.
- `D′u`: Cache array for computing `D′(u)`.
- `Ru`: Cache array for computing `R(u)`.
- `R′u`: Cache array for computing `R′(u)`.
- `A`: Matrix for the linear system giving the coefficients.
"""
function basis_bootstrap_helper(x, t, bootₓ, bootₜ, d, r, B)
    # Compute the extrema and range
    x_min, x_max = extrema(x)
    t_min, t_max = extrema(t)
    x_rng = x_max - x_min
    t_rng = t_max - t_min

    # Compute the grid
    nₓ = length(bootₓ)
    nₜ = length(bootₜ)
    x̃ = repeat(bootₓ, outer = nₜ)
    t̃ = repeat(bootₜ, inner = nₓ)

    # Scale the vectors 
    @. x̃ = (x̃ - x_min) / x_rng
    @. t̃ = (t̃ - t_min) / t_rng

    # Compute the test matrix 
    Xₛ = [vec(x̃)'; vec(t̃)']
    nₓnₜ = size(Xₛ, 2)

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
    diffusionBases = zeros(d, B)
    reactionBases = zeros(r, B)

    # Drawing samples
    ℓz = zeros(4nₓnₜ) # Storing the matrix-vector product Lz
    zvals = zeros(4nₓnₜ, B) # Storing the random standard normal variables

    # Caches 
    A = zeros(nₓnₜ, d + r)

    # Return 
    return x_min, t_min, x_rng, t_rng, Xₛ,
    f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
    diffusionBases, reactionBases,
    ℓz, zvals,
    A
end

"""
    basis_learn_equations!(f, fₜ, fₓ, fₓₓ, D, D′, R, 
        db, rb, d, r, A, 
        D_params, R_params, inIdx)

Estimates the coefficients `db` and `rb` for diffusion and reaction.

# Arguments 
- `f`: Estimates for the learned function `f` of the reaction-diffusion process.
- `fₓ`: Derivatives of `f` in `x` at the same gridpoints.
- `fₓₓ`: Second derivatives of `f` in `x` at the same gridpoints.
- `fₜ`: Derivatives of `f` in `t` at the same gridpoints.
- `D`: The basis functions for the diffusion curve, provided as a vector of functions.
- `D′`: The derivatives for the basis functions `φ`, provided as a vector of functions. (These are `φ′ = d/du[φ(u)]`, so do not divide these by K.)
- `R`: The basis functions for the reaction curve, provided as a vector of functions.
- `db`: Cache array for the basis coefficients for diffusion.
- `rb`: Cache array for the basis coefficients for reaction.
- `d`: Number of diffusion parameters.
- `r`: Number of reaction parameters.
- `A`: Cache matrix for the coefficient matrix.
- `D_params`: Known diffusion parameters. 
- `R_params`: Known reaction parameters.
- `inIdx`: Values to use. See [`data_thresholder`](@ref).

# Outputs 
`db` and `rb` are updated in-place with the diffusion and reaction coefficients, respectively.
"""
function basis_learn_equations!(f, fₜ, fₓ, fₓₓ, D, D′, R,
    db, rb, d, r, A,
    D_params, R_params, inIdx)
    @inbounds @views for j = 1:d # Use @inbounds to tell Julia that all calls to getindex are in the array's size. Use @muladd to help detect whether fused multiply-add is OK. Use @. to distribute broadcasts easily. Use @views to convert all array calls into view calls.
        A[:, j] .= D′[j].(f, Ref(D_params)) .* (fₓ.^2) .+ D[j].(f, Ref(D_params)) .* fₓₓ
    end
    @inbounds @views for j = 1:r
        A[:, d+j] .= R[j].(f, Ref(R_params))
    end
    soln = @views A[inIdx, :] \ fₜ[inIdx]
    db .= @view soln[1:d]
    rb .= @view soln[(d+1):(d+r)]
    return nothing
end

"""
    basis_bootstrap_gp(x::T1, t::T1, u::T1,
        D::Vector{Function}, D′::Vector{Function}, R::Vector{Function}, R′::Vector{Function};
        gp_setup::GP_Setup = GP_Setup(u),
        bootstrap_setup::Bootstrap_Setup = Bootstrap_Setup(x, t, u),
        optim_setup::Optim.Options = Optim.Options(),
        pde_setup::PDE_Setup = PDE_Setup(x),
        D_params = nothing, R_params = nothing, PDEkwargs...) where {T1<:AbstractVector}

Perform bootstrapping on the data `(x, t, u)` to learn the appropriate functional forms of 
`(T, D, R)` with uncertainty, using the basis function approach.

# Arguments 
- `x::T1`: The spatial data.
- `t::T1`: The temporal data.
- `u::T1`: The density data.
- `D::Vector{Function}`: The diffusion function, given in the form `D(u, β, D_params)`.
- `D′::Vector{Function}`: The derivative of the diffusion function, given in the form `D′(u, β, D_params)`.
- `R::Vector{Function}`: The reaction function, given in the form `R(u, γ, R_params)`.
- `R′::Vector{Function}`: The derivative of the reaction function, given in the form `R′(u, γ, R_params)`.

# Keyword Arguments 
- `gp_setup::GP_Setup = GP_Setup(u)`: Defines the setup for the Gaussian process. See also [`GP_Setup`](@ref).
- `bootstart_setup::Bootstrap_Setup = Bootstrap_Setup(x, t, u)`: Defines some extra keyword arguments for the bootstrapping and optimisation process. See also [`Bootstrap_Setup`](@ref).
- `pde_setup::PDE_Setup = PDE_Setup(x)`: Defines some extra keyword arguments for the PDE solutions. See also [`PDE_Setup`](@ref).
- `D_params = nothing`: Extra known parameters for the diffusion function `D`.
- `R_params = nothing`: Extra known parameters for the reaction function `R`.
- `PDEkwargs...`: Extra keyword arguments to use inside `DifferentialEquations.solve`.

# Outputs
- `bgp`: A `BasisBootResults` structure. See [`BasisBootResults`](@ref). 
"""
function basis_bootstrap_gp(x::T1, t::T1, u::T1,
    D::Vector{Function}, D′::Vector{Function}, R::Vector{Function}, R′::Vector{Function};
    gp_setup::GP_Setup = GP_Setup(u),
    bootstrap_setup::Bootstrap_Setup = Bootstrap_Setup(x, t, u),
    pde_setup::PDE_Setup = PDE_Setup(x),
    D_params = nothing, R_params = nothing, PDEkwargs...) where {T1<:AbstractVector}
    ## Check provided functions and ODE algorithm are correct
    #@assert length(x) == length(t) == length(u) "The lengths of the provided data vectors must all be equal."
    d = length(D)
    r = length(R)
    try
        evaluate_basis(ones(d), D, u[1], D_params)
    catch
        throw("Error evaluating the diffusion function. Check that D_params = $D_params has not been incorrectly specified.")
    end
    try
        evaluate_basis(ones(r), R, u[1], R_params)
    catch
        throw("Error evaluating the reaction function. Check that R_params = $R_params has not been incorrectly specified.")
    end

    ## Fit the GP 
    gp = ismissing(gp_setup.gp) ? fit_GP(x, t, u, gp_setup) : gp_setup.gp

    ## Setup the bootstrapping grid, define some parameters, construct cache arrays, etc.
    x_min, t_min, x_rng, t_rng, Xₛ,
    f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
    diffusionBases, reactionBases,
    ℓz, zvals,
    A = basis_bootstrap_helper(x, t, bootstrap_setup.bootₓ, bootstrap_setup.bootₜ, d, r, bootstrap_setup.B)

    ## Compute the mean vector and Cholesky factor for the joint Gaussian process of the function and its derivatives 
    if ismissing(gp_setup.μ) || ismissing(gp_setup.L)
        μ, L = compute_joint_GP(gp, Xₛ; nugget = gp_setup.nugget)
    else
        μ, L = gp_setup.μ, gp_setup.L
    end

    ## Now do the equation learning 
    j = 1
    while j ≤ bootstrap_setup.B
        # Draw from N(0, 1) for sampling from the Gaussian process 
        @views randn!(zvals[:, j])

        # Compute the required functions and derivatives 
        @views draw_gp!(ffₜfₓfₓₓ, μ, L, zvals[:, j], ℓz)
        f .= ffₜfₓfₓₓ[f_idx]
        fₜ .= ffₜfₓfₓₓ[fₜ_idx] / t_rng
        fₓ .= ffₜfₓfₓₓ[fₓ_idx] / x_rng
        fₓₓ .= ffₜfₓfₓₓ[fₓₓ_idx] / x_rng^2

        # Threshold the data 
        inIdx = data_thresholder(f, fₜ, bootstrap_setup.τ)

        ## Parameter estimation 
        @views basis_learn_equations!(f, fₜ, fₓ, fₓₓ, D, D′, R,
            diffusionBases[:, j], reactionBases[:, j], d, r, A,
            D_params, R_params, inIdx)

        if @views !any(isnan.(reactionBases[:, j])) && !any(isnan.(diffusionBases[:, j]))
            j += 1
            print("Bootstrapping: Step $j of $(bootstrap_setup.B).\u001b[1000D")
        end
    end

    Xₛⁿ = deepcopy(Xₛ)
    @views @. Xₛⁿ[1, :] = Xₛ[1, :] * x_rng + x_min
    @views @. Xₛⁿ[2, :] = Xₛ[2, :] * t_rng + t_min

    ## Return the results 
    return BasisBootResults(diffusionBases, reactionBases, gp, zvals, Xₛ, Xₛⁿ, bootstrap_setup.bootₓ, bootstrap_setup.bootₜ, D, D′, R, R′, D_params, R_params, μ, L, gp_setup, bootstrap_setup, pde_setup)
end

"""
    basis_sysdegeneral!(dudt, u, p, t)

Function for computing the system of ODEs used in a discretised delay-reaction-diffusion PDE.

# Arguments 
- `dudt`: A cache array used for storing the left-hand side of the system of ODEs. 
- `u`: The current values for the variables in the systems.
- `p`: A tuple of parameters, given by:
    - `p[1] = N`: Number of mesh points being used for solving the PDE.
    - `p[2] = V`: The volume of each cell in the discretised PDE.
    - `p[3] = h`: The spacing between cells in the discretised PDE.
    - `p[4] = a₀`: The coefficient on `u(a, t)` in the Robin boundary condition at `x = a` (the left end-point of the mesh). 
    - `p[5] = b₀`: The coefficient on `-∂u(a, t)/∂x` in the Robin boundary condition at `x = a` (the left end-point of the mesh).
    - `p[6] = c₀`: The right-hand side constant in the Robin boundary condition at `x = a` (the left end-point of the mesh).
    - `p[7] = a₁`: The coefficient on `u(b, t)` in the Robin boundary condition at `x = b` (the right end-point of the mesh). 
    - `p[8] = b₁`: The coefficient on `∂u(b, t)/∂t` in the Robin boundary condition at `x = b` (the right end-point of the mesh).
    - `p[9] = c₁`: The right-hand side constant in the Robin boundary condition at `x = b` (the right end-point of the mesh).
    - `p[10] = DD`: Cache array used for storing the values of the diffusion function `D` at `D(u)`.
    - `p[11] = RR`: Cache array used for storing the values of the reaction function `R` at `R(u)`.
    - `p[12] = DD′`: Cache array used for storing the values of the derivative of the diffusion function `D` at `D′(u)`.
    - `p[13] = RR′`: Cache array used for storing the values of the derivative of the reaction function `R` at `R′(u)`.
    - `p[14] = D`: The diffusion function, given in the form `D(u, β, D_params)`.
    - `p[15] = R`: The reaction function, given in the form `R(u, γ, R_params)`.
    - `p[16] = D′`: The derivative of the diffusion function, given in the form `D′(u, β, D_params)`.
    - `p[17] = R′`: The derivative of the reaction function, given in the form `R′(u, γ, R_params)`.
    - `p[18] = db`: The values of the diffusion parameters.
    - `p[19] = rb`: The values of the reaction parameters.
    - `p[20] = D_params`: Extra parameters used in the diffusion function.
    - `p[21] = R_params`: Extra parameters for the reaction function.
    - `p[22] = A₁`: A cache array used when computing the values of the diffusion function at a point u. Must be the same length as `meshPoints`.
    - `p[23] = A₂`: A cache array used when computing the values of the reaction function at a point u. Must be the same length as `meshPoints`.
- `t`: The current time value.

# Outputs 
The values are updated in-place into the vector `dudt` for the new value of `dudt` at time `t`.
"""
function basis_sysdegeneral!(dudt, u, p, t)
    N, V, h, a₀, b₀, c₀, a₁, b₁, c₁, DD, RR, DD′, RR′, D, R, D′, R′, db, rb, D_params, R_params, A₁, A₂ = p
    evaluate_basis!(DD, db, D, u, D_params, A₁)
    evaluate_basis!(RR, rb, R, u, R_params, A₂)
    @muladd @inbounds begin
        dudt[1] = 1.0 / V[1] * ((DD[1] + DD[2]) / 2.0 * ((u[2] - u[1]) / h[1]) - a₀ / b₀ * DD[1] * u[1]) + c₀ / (b₀ * V[1]) * DD[1] + RR[1]
        for i = 2:(N-1)
            dudt[i] = 1 / V[i] * ((DD[i] + DD[i+1]) / 2.0 * ((u[i+1] - u[i]) / h[i]) - ((DD[i-1] + DD[i])) / 2.0 * ((u[i] - u[i-1]) / h[i-1])) + RR[i]
        end
        dudt[N] = 1.0 / V[N] * (-a₁ / b₁ * DD[N] * u[N] - ((DD[N-1] + DD[N]) / 2.0) * ((u[N] - u[N-1]) / h[N-1])) + c₁ / (b₁ * V[N]) * DD[N] + RR[N]
    end
    return nothing
end

"""
    compute_valid_pde_indices(bgp, u_pde, num_t, num_u, B, tr, dr, rr, nodes, weights, D_params, R_params, T_params)

Computes the indices corresponding to the bootstrap samples which give valid PDE solutions. The check is done by 
ensuring that the delay and diffusion values are strictly nonnegative, and the area under the reaction curve is nonnegative.

# Arguments 
- `bgp::BasisBootResults`: The bootstrapping results.
- `u_pde`: The density data used for fitting the original Gaussian process. See also [`fit_GP`](@ref).
- `num_u`: The number of density values to use for checking the validity of the diffusion function values.
- `B`: The number of bootstrap samples used.
- `dr`: The matrix of estimated diffusion parameters.
- `rr`: The matrix of estimated reaction parameters.
- `nodes`: Gauss-Legendre quadrature nodes.
- `weights`: Gauss-Legendre quadrature weights.
- `D_params`: Extra parameters for the diffusion function.
- `R_params`: Extra parameters for the reaction function. 

# Outputs 
- `idx`: The vector of indices corresponding to valid bootstrap samples.
"""
function compute_valid_pde_indices(bgp::BasisBootResults, u_pde, num_u, B, dr, rr, nodes, weights, D_params, R_params)
    idx = Array{Int64}(undef, 0)
    u_vals = range(minimum(bgp.gp.y), maximum(bgp.gp.y), length = num_u)
    Duv = zeros(num_u, 1)
    max_u = maximum(u_pde)
    A₁ = zeros(num_u, length(bgp.D))
    @inbounds @views for j = 1:B
        evaluate_basis!(Duv, dr[:, j], bgp.D, u_vals, D_params, A₁)
        Reaction = u -> evaluate_basis(rr[:, j], bgp.R, max_u / 2 * (u + 1), R_params) # missing a max_u/2 factor in front for this new integral, but thats fine since it doesn't change the sign
        Ival = dot(weights, Reaction.(nodes))
        if all(≥(0), Duv) && Ival[1] ≥ 0.0 # Safeguard... else the solution never finishes!
            push!(idx, j)
        end
    end
    return idx
end

"""
    compute_valid_pde_indices(u_pde, num_t, num_u, nodes, weights, bgp::BasisBootResults)

Method for calling [`compute_valid_pde_indices`] when providing only `bgp` and the data. 
"""
function compute_valid_pde_indices(u_pde, num_u, nodes, weights, bgp::BasisBootResults)
    return compute_valid_pde_indices(bgp, u_pde, num_u, bgp.bootstrap_setup.B, bgp.diffusionBases, bgp.reactionBases, nodes, weights, bgp.D_params, bgp.R_params)
end

"""
    boot_pde_solve(bgp::BasisBootResults, x_pde, t_pde, u_pde; prop_samples = 1.0, ICType = "data")

Solve the PDEs corresponding to the bootstrap iterates in `bgp` obtained from [`basis_bootstrap_gp`](@ref). 

# Arguments 
- `bgp::BasisBootResults`: A [`BasisBootResults`](@ref) struct containing the results from [`basis_bootstrap_gp`](@ref).
- `x_pde`: The spatial data to use for obtaining the initial condition.
- `t_pde`: The temporal data to use for obtaining the initial condition.
- `u_pde`: The density data to use for obtaining the initial condition.

# Keyword Arguments 
- `prop_samples = 1.0`: The proportion of bootstrap samples to compute teh corresponding PDE soluton to.
- `ICType = "data"`: The type of initial condition to use. Should be either `"data"` or `"gp"`.

# Outputs 
- `solns_all`: The solutions to the PDEs over the mesh points at each time value.

# Note 
The `_pde` subscript is used to indicate that these data need not be the same as the `(x, t, u)` used in [`bootstrap_gp`](@ref), for example.
For example, we may have 3 replicates of some data which we would easily use in [`bootstrap_gp`](@ref), but for the PDE we would need to average these 
together for obtaining the solutions.
"""
function boot_pde_solve(bgp::BasisBootResults, x_pde, t_pde, u_pde; prop_samples = 1.0, ICType = "data")
    #@assert 0 < prop_samples ≤ 1.0 "The values of prop_samples must be in (0, 1]."
    #@assert ICType ∈ ["data", "gp"]
    nodes, weights = gausslegendre(5)
    d = length(bgp.D)
    r = length(bgp.R)
    # Compute number of bootstrap replicates 
    dr = bgp.diffusionBases
    rr = bgp.reactionBases
    B = size(dr, 2)

    # Setup PDE
    rand_pde = convert(Int64, trunc(prop_samples * B))
    N = length(bgp.pde_setup.meshPoints)
    M = length(bgp.pde_setup.δt)
    solns_all = zeros(N, rand_pde, M)

    # Compute the initial conditions
    initialCondition_all = compute_initial_conditions(x_pde, t_pde, u_pde, bgp, ICType)

    # Find the valid indices
    idx = compute_valid_pde_indices(u_pde, 500, nodes, weights, bgp)

    # Solve PDEs
    initialCondition = zeros(N, 1) # Setup cache array
    finalTime = maximum(bgp.pde_setup.δt)
    sample_idx = StatsBase.sample(idx, rand_pde)
    Du = zeros(N, 1)
    Ru = zeros(N, 1)
    D′u = zeros(N, 1)
    R′u = zeros(N, 1)
    A₁ = zeros(N, d)
    A₂ = zeros(N, r)
    Δx = diff(bgp.pde_setup.meshPoints)
    V = @views 1 / 2 * [Δx[1]; Δx[1:(N-2)] + Δx[2:(N-1)]; Δx[N-1]]
    tspan = (0.0, finalTime)
    @views @muladd @inbounds for (pdeidx, coeff) in collect(enumerate(sample_idx)) # Don't need collect here, but it was useful previously when dealing with threading (before we removed threading). Just keeping it as a reminder.
        initialCondition .= initialCondition_all[:, coeff]
        p = (N, V, Δx, bgp.pde_setup.LHS..., bgp.pde_setup.RHS..., Du, Ru, D′u, R′u, bgp.D, bgp.R, bgp.D′, bgp.R′, dr[:, coeff], rr[:, coeff], bgp.D_params, bgp.R_params, A₁, A₂)
        prob = ODEProblem(basis_sysdegeneral!, initialCondition, tspan, p)
        #try
            solns_all[:, pdeidx, :] .= hcat(DifferentialEquations.solve(prob, bgp.pde_setup.alg, saveat = bgp.pde_setup.δt).u...)
            print("Solving PDEs: Step $pdeidx of $rand_pde.\u001b[1000D") # https://discourse.julialang.org/t/update-variable-in-logged-message-without-printing-a-new-line/32755
        #catch
        #end
    end

    # Return
    return solns_all
end

"""
    density_values(bgp::BasisBootResults; <keyword arguments>)

Computes the densities for the bootstrapping results in `bgp` from a basis function approach.

# Arguments 
- `bgp`: A [`BasisBootResults`](@ref) object which contains the results for the bootstrapping. 

# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 23`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `diffusion_scales = nothing`: Values that multiply the individual diffusion parameters. 
- `reaction_scales = nothing`: Values that multiply the individual reaction parameters.

# Outputs 
- `dr`: Diffusion densities. 
- `rr`: Reaction densities. 
- `d`: Number of diffusion parameters.
- `r`: Number of reaction parameters. 
- `delayCIs`: Confidence intervals for the delay parameters. 
- `diffusionCIs`: Confidence intervals for the diffusion parameters. 
- `reactionCIS`: Confidence intervals for the reaction parameters.
"""
function density_values(bgp::BasisBootResults; level = 0.05, diffusion_scales = nothing, reaction_scales = nothing)
    # Work on diffusion 
    quantiles = [level / 2 1 - level / 2]
    dr = copy(bgp.diffusionBases)
    if !isnothing(diffusion_scales)
        try
            dr .*= vec(diffusion_scales)
        catch
            dr = dr * diffusion_scales
        end
    end
    d = size(dr, 1)
    diffusionCIs = zeros(d, 2)
    @inbounds @views for j = 1:d
        diffusionCIs[j, :] .= quantile(dr[j, :], quantiles)[1:2]
    end

    # Now do reaction 
    rr = copy(bgp.reactionBases)
    if !isnothing(reaction_scales)
        try
            rr .*= vec(reaction_scales)
        catch
            rr = rr * reaction_scales
        end
    end
    r = size(rr, 1)
    reactionCIs = zeros(r, 2)
    @inbounds @views for j = 1:r
        reactionCIs[j, :] .= quantile(rr[j, :], quantiles)[1:2]
    end
    return dr, rr, d, r, diffusionCIs, reactionCIs
end

"""
    density_results(bgp::BasisBootResults; <keyword arguments>)

Plots the densities for the bootstrapping results in `bgp` with the basis function approach.

# Arguments 
- `bgp`: A [`BasisBootResults`](@ref) object which contains the results for the bootstrapping. 

# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 23`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `diffusion_scales = nothing`: Values that multiply the individual diffusion parameters. 
- `reaction_scales = nothing`: Values that multiply the individual reaction parameters.
- `diffusion_resolution = (800, 800)`: Resolution for the diffusion figure.
- `reaction_resolution = (800, 800)`: Resolution for the reaction figure.

# Outputs 
- `diffusionDensityFigure`: A figure of plots containing a density plot for each diffusion parameter.
- `reactionDensityFigure`: A figure of plots containing a density plot for each reaction parameter.
"""
function density_results(bgp::BasisBootResults; level = 0.05, fontsize = 23, diffusion_scales = nothing, reaction_scales = nothing,
    diffusion_resolution = (800, 800), reaction_resolution = (800, 800))

    # Compute densities 
    dr, rr, d, r, diffusionCIs, reactionCIs = density_values(bgp; level = level, diffusion_scales = diffusion_scales, reaction_scales = reaction_scales)
    # Pre-allocate the plots 
    diffusionDensityAxes = Vector{Axis}(undef, d)
    reactionDensityAxes = Vector{Axis}(undef, r)
    alphabet = join('a':'z') # For labelling the figures

    # Plot the diffusion coefficient densities 
    diffusionDensityFigure = Figure(fontsize = fontsize, resolution = diffusion_resolution)
    for i = 1:d
        diffusionDensityAxes[i] = Axis(diffusionDensityFigure[1, i], xlabel = L"\beta_%$i", ylabel = "Probability density",
            title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], diffusionCIs[i, 1], diffusionCIs[i, 2]),
            titlealign = :left)
        densdat = KernelDensity.kde(dr[i, :])
        lines!(diffusionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
        CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
        band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
    end

    # Plot the reaction coefficient densities
    reactionDensityFigure = Figure(fontsize = fontsize, resolution = reaction_resolution)
    for i = 1:r
        reactionDensityAxes[i] = Axis(reactionDensityFigure[1, i], xlabel = L"\gamma_%$i", ylabel = "Probability density",
            title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], reactionCIs[i, 1], reactionCIs[i, 2]),
            titlealign = :left)
        densdat = kde(rr[i, :])
        lines!(reactionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
        CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
        band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
    end

    # Return
    return diffusionDensityFigure, reactionDensityFigure
end

"""
    curve_values(bgp::BasisBootResults; <keyword arguments>)

Computes values for plotting the learned functional forms along with confidence intervals for the bootstrapping results in `bgp` from the basis function approach.

# Arguments 
- `bgp`: A [`BasisBootResults`](@ref) object which contains the results for the bootstrapping. 
    
# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `x_scale = 1.0`: Value used for scaling the spatial data (and all other length units, e.g. for diffusion).
- `t_scale = 1.0`: Value used for scaling the temporal data (and all other time units, e.g. for reaction).

# Outputs
- `Du_vals`: Ribbon features for the diffusion functions.
- `Ru_vals`: Ribbon features for the reaction functions.
- `u_vals`: The density values used for computing the functions. 
- `t_vals`: The time values used for computing the functions.
"""
function curve_values(bgp::BasisBootResults; level = 0.05, x_scale = 1.0, t_scale = 1.0)
    # Setup parameters and grid for evaluation
    dr = bgp.diffusionBases
    rr = bgp.reactionBases
    B = size(dr, 2)
    num_u = 500
    u_vals = collect(range(minimum(bgp.gp.y), maximum(bgp.gp.y), length = num_u))

    # Evaluate curves 
    Du = zeros(num_u, B)
    Ru = zeros(num_u, B)
    @views @inbounds for j = 1:B
        Du[:, j] .= evaluate_basis.(Ref(dr[:, j]), Ref(bgp.D), u_vals, Ref(bgp.D_params)) * x_scale^2 / t_scale
        Ru[:, j] .= evaluate_basis.(Ref(rr[:, j]), Ref(bgp.R), u_vals, Ref(bgp.R_params)) / t_scale
    end

    # Find lower/upper values for confidence intervals, along with mean curves
    Du_vals = compute_ribbon_features(Du; level = level)
    Ru_vals = compute_ribbon_features(Ru; level = level)

    # Return 
    return Du_vals, Ru_vals, u_vals
end

"""
    curve_results(bgp::BasisBootResults; <keyword arguments>)

Plots the learned functional forms along with confidence intervals for the bootstrapping results in `bgp` from the basis function approach.

# Arguments 
- `bgp`: A [`BasisBootResults`](@ref) object which contains the results for the bootstrapping. 
    
# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 23`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `x_scale = 1.0`: Value used for scaling the spatial data (and all other length units, e.g. for diffusion).
- `t_scale = 1.0`: Value used for scaling the temporal data (and all other time units, e.g. for reaction).

# Outputs
- `diffusionCurvePlots`: A plot containing the learned functional form for the diffusion function, along with an uncertainty ribbon.
- `reactionCurvePlots`: A plot containing the learned functional form for the reaction function, along with an uncertainty ribbon.
"""
function curve_results(bgp::BasisBootResults; level = 0.05, fontsize = 23, x_scale = 1.0, t_scale = 1.0)
    # Compute values 
    Du_vals, Ru_vals, u_vals = curve_values(bgp; level = level, x_scale = x_scale, t_scale = t_scale)

    # Plot the diffusion curves 
    diffusionCurvePlots = Figure(fontsize = fontsize)
    ax = Axis(diffusionCurvePlots[1, 1], xlabel = L"u", ylabel = L"D(u)", linewidth = 1.3, linecolor = :blue)
    lines!(ax, u_vals / x_scale^2, Du_vals[1])
    band!(ax, u_vals / x_scale^2, Du_vals[3], Du_vals[2], color = (:blue, 0.35))

    # Plot the reaction curves 
    reactionCurvePlots = Figure(fontsize = fontsize)
    ax = Axis(reactionCurvePlots[1, 1], xlabel = L"u", ylabel = L"R(u)", linewidth = 1.3, linecolor = :blue)
    lines!(ax, u_vals / x_scale^2, Ru_vals[1])
    band!(ax, u_vals / x_scale^2, Ru_vals[3], Ru_vals[2], color = (:blue, 0.35))

    # Return 
    return diffusionCurvePlots, reactionCurvePlots
end

