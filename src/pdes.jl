#####################################################################
## Script description: pdes.jl 
##
## This script contains certain functions used for working with the 
## partial differential equations used with bootstrapping and computing 
## results.
##
## The following functions are defined:
##  - sysdegeneral!: Computes the system of ODEs in a discretised
##      delay-reaction-diffusion PDE.
##  - compute_initial_conditions: Computes initial conditions to use.
##  - compute_valid_pde_indices: Checks which bootstrap iterates will 
##      lead to a solution that exists.
##  - boot_pde_solve: Solve the PDEs for each bootstrap iterate.
##
#####################################################################

"""
    sysdegeneral!(dudt, u, p, t)

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
    - `p[12] = T`: The delay function, given in the form `T(t, α, T_params)`.
    - `p[13] = D`: The diffusion function, given in the form `D(u, β, D_params)`.
    - `p[14] = R`: The reaction function, given in the form `R(u, γ, R_params)`.
    - `p[15] = tb`: The values of the delay parameters.
    - `p[16] = db`: The values of the diffusion parameters.
    - `p[17] = rb`: The values of the reaction parameters.
    - `p[18] = D_params`: Extra parameters used in the diffusion function.
    - `p[19] = R_params`: Extra parameters for the reaction function.
    - `p[20] = T_params`: Extra parameters for the delay function.
- `t`: The current time value.

# Outputs 
The values are updated in-place into the vector `dudt` for the new value of `dudt` at time `t`.
"""
function sysdegeneral!(dudt, u, p, t)
    N, V, h, a₀, b₀, c₀, a₁, b₁, c₁, DD, RR, T, D, R, tb, db, rb, D_params, R_params, T_params = p
    if typeof(DD) <: PreallocationTools.DiffCache # If we're doing automatic differentiation 
        DD = get_tmp(DD, D(u[1], db, D_params))
        RR = get_tmp(RR, R(u[1], rb, R_params))
    end
    for (j, uval) in enumerate(u)
        DD[j] = D(uval, db, D_params)
        RR[j] = R(uval, rb, R_params)
    end
    Tt = T(t, tb, T_params)
    @muladd @inbounds begin
        dudt[1] = 1.0 / V[1] * ((DD[1] + DD[2]) / 2.0 * ((u[2] - u[1]) / h[1]) - a₀ / b₀ * DD[1] * u[1]) + c₀ / (b₀ * V[1]) * DD[1] + RR[1]
        for i = 2:(N-1)
            dudt[i] = 1 / V[i] * ((DD[i] + DD[i+1]) / 2.0 * ((u[i+1] - u[i]) / h[i]) - ((DD[i-1] + DD[i])) / 2.0 * ((u[i] - u[i-1]) / h[i-1])) + RR[i]
        end
        dudt[N] = 1.0 / V[N] * (-a₁ / b₁ * DD[N] * u[N] - ((DD[N-1] + DD[N]) / 2.0) * ((u[N] - u[N-1]) / h[N-1])) + c₁ / (b₁ * V[N]) * DD[N] + RR[N]
    end
    dudt .*= Tt # Delay
    return nothing
end

"""
    compute_initial_conditions(x_pde, t_pde, u_pde, ICType, bgp::BootResults, N, B, meshPoints)

Computes initial conditions for the bootstrap iterates in `bgp`. See also [`bootstrap_gp`](@ref).

# Arguments 
- `x_pde`: The spatial data used for fitting the original Gaussian process. See also [`fit_GP`](@ref).
- `t_pde`: The temporal data used for fitting the original Gaussian process. See also [`fit_GP`](@ref).
- `u_pde`: The density data used for fitting the original Gaussian process. See also [`fit_GP`](@ref).
- `ICType`: The type of initial condition to use. If `ICType == "data"` then the initial condition is simply a spline through the data, and if `ICType == "gp"` then the initial condition is a sample of the underlying Gaussian process in `bgp.gp`.
- `bgp::BootResults`: The bootstrapping results of type [`BootResults`](@ref). See [`bootstrap_gp`](@ref).
- `N`: The number of mesh points.
- `B`: The number of bootstrap iterations.
- `meshPoints`: The spatial mesh.

# Outputs 
- `initialCondition_all`: The initial condition to use for each bootstrap iterate, with the `j`th column corresponding to the `j`th bootstrap sample.
"""
function compute_initial_conditions(x_pde, t_pde, u_pde, ICType, bgp::BootResults, N, B, meshPoints)
    @assert ICType ∈ ["data", "gp"] "The provided ICType must be either \"data\" or \"gp\"."
    initialCondition_all = zeros(N, B)
    @views if ICType == "data"
        position₁ = x_pde[t_pde.==0.0]
        u0 = u_pde[t_pde.==0.0]
        initialCondition_all .= reshape(repeat(Dierckx.Spline1D(position₁, u0; k = 1)(meshPoints), outer = B), (N, B)) # Same initial condition for all B replicates 
    elseif ICType == "gp"
        nₓnₜ = convert(Int64, length(bgp.μ) / 4)
        bigf = (bgp.μ.+bgp.L*bgp.zvals)[1:nₓnₜ, :] # Extract u from [f;∂ₜf;∂ₓf;∂ₓₓf]
        initialCondition_all_nospline = bigf[bgp.Xₛ[2, :].==0.0, :] |> x -> max.(x, 0.0) # Compute initial conditions for all sampled curves, then make them non-negative
        for j = 1:B
            initialCondition_all[:, j] .= Dierckx.Spline1D(bgp.bootₓ, initialCondition_all_nospline[:, j]; k = 1)(meshPoints)
        end
    end
    return initialCondition_all
end

"""
    compute_initial_conditions(x_pde, t_pde, u_pde, bgp::BootResults)

Method for calling [`compute_initial_conditions`] when providing only `bgp` and the data. 
"""
function compute_initial_conditions(x_pde, t_pde, u_pde, bgp::BootResults, ICType)
    return compute_initial_conditions(x_pde, t_pde, u_pde, ICType, bgp, length(bgp.pde_setup.meshPoints), bgp.bootstrap_setup.B, bgp.pde_setup.meshPoints)
end

"""
    compute_valid_pde_indices(bgp, u_pde, num_t, num_u, B, tr, dr, rr, nodes, weights, D_params, R_params, T_params)

Computes the indices corresponding to the bootstrap samples which give valid PDE solutions. The check is done by 
ensuring that the delay and diffusion values are strictly nonnegative, and the area under the reaction curve is nonnegative.

# Arguments 
- `bgp::BootResults`: The bootstrapping results of type [`BootResults`](@ref). See [`bootstrap_gp`](@ref).
- `u_pde`: The density data used for fitting the original Gaussian process. See also [`fit_GP`](@ref).
- `num_t`: The number of time values to use for checking the validity of the delay function values. 
- `num_u`: The number of density values to use for checking the validity of the diffusion function values.
- `B`: The number of bootstrap samples used.
- `tr`: The matrix of estimated delay parameters. 
- `dr`: The matrix of estimated diffusion parameters.
- `rr`: The matrix of estimated reaction parameters.
- `nodes`: Gauss-Legendre quadrature nodes.
- `weights`: Gauss-Legendre quadrature weights.
- `D_params`: Extra parameters for the diffusion function.
- `R_params`: Extra parameters for the reaction function. 
- `T_params`: Extra parameters for the delay function.

# Outputs 
- `idx`: The vector of indices corresponding to valid bootstrap samples.
"""
function compute_valid_pde_indices(bgp::BootResults, u_pde, num_t, num_u, B, tr, dr, rr, nodes, weights, D_params, R_params, T_params)
    idx = Array{Int64}(undef, 0)
    u_vals = range(minimum(bgp.gp.y), maximum(bgp.gp.y), length = num_u)
    t_vals = collect(range(minimum(bgp.Xₛⁿ[2, :]), maximum(bgp.Xₛⁿ[2, :]), length = num_t))
    Tuv = zeros(num_t, 1)
    Duv = zeros(num_u, 1)
    max_u = maximum(u_pde)
    @inbounds @views for j = 1:B
        Duv .= bgp.D.(u_vals, Ref(dr[:, j]), Ref(D_params))
        Tuv .= bgp.T.(t_vals, Ref(tr[:, j]), Ref(T_params))
        Reaction = u -> bgp.R(max_u / 2 * (u + 1), rr[:, j], R_params) # missing a max_u/2 factor in front for this new integral, but thats fine since it doesn't change the sign
        Ival = dot(weights, Reaction.(nodes))
        if all(≥(0), Tuv) && all(≥(0), Duv) && Ival[1] ≥ 0.0 # Safeguard... else the solution never finishes!
            push!(idx, j)
        end
    end
    return idx
end

"""
    compute_valid_pde_indices(u_pde, num_t, num_u, nodes, weights, bgp::BootResults)

Method for calling [`compute_valid_pde_indices`] when providing only `bgp` and the data. 
"""
function compute_valid_pde_indices(u_pde, num_t, num_u, nodes, weights, bgp::BootResults)
    return compute_valid_pde_indices(bgp, u_pde, num_t, num_u, bgp.bootstrap_setup.B, bgp.delayBases, bgp.diffusionBases, bgp.reactionBases, nodes, weights, bgp.D_params, bgp.R_params, bgp.T_params)
end

"""
    boot_pde_solve(bgp::BootResults, x_pde, t_pde, u_pde; prop_samples = 1.0, ICType = "data")

Solve the PDEs corresponding to the bootstrap iterates in `bgp` obtained from [`bootstrap_gp`](@ref). 

# Arguments 
- `bgp::BootResults`: A [`BootResults`](@ref) struct containing the results from [`bootstrap_gp`](@ref).
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
function boot_pde_solve(bgp::BootResults, x_pde, t_pde, u_pde; prop_samples = 1.0, ICType = "data")
    todo"Do some further analysis to find the best ODE algorithm to use."
    @assert 0 < prop_samples ≤ 1.0 "The values of prop_samples must be in (0, 1]."
    @assert ICType ∈ ["data", "gp"]
    nodes, weights = gausslegendre(5)
    # Compute number of bootstrap replicates 
    tr = bgp.delayBases
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
    idx = compute_valid_pde_indices(u_pde, 500, 500, nodes, weights, bgp)

    # Solve PDEs
    initialCondition = zeros(N, 1) # Setup cache array
    finalTime = maximum(bgp.pde_setup.δt)
    sample_idx = StatsBase.sample(idx, rand_pde)
    Du = zeros(N, 1)
    Ru = zeros(N, 1)
    Δx = diff(bgp.pde_setup.meshPoints)
    V = @views 1 / 2 * [Δx[1]; Δx[1:(N-2)] + Δx[2:(N-1)]; Δx[N-1]]
    tspan = (0.0, finalTime)
    @views @muladd @inbounds for (pdeidx, coeff) in collect(enumerate(sample_idx)) # Don't need collect here, but it was useful previously when dealing with threading (before we removed threading). Just keeping it as a reminder.
        initialCondition .= initialCondition_all[:, coeff]
        p = (N, V, Δx, bgp.pde_setup.LHS..., bgp.pde_setup.RHS..., Du, Ru, bgp.T, bgp.D, bgp.R, tr[:, coeff], dr[:, coeff], rr[:, coeff], bgp.D_params, bgp.R_params, bgp.T_params)
        prob = ODEProblem(sysdegeneral!, initialCondition, tspan, p)
        solns_all[:, pdeidx, :] .= hcat(DifferentialEquations.solve(prob, bgp.pde_setup.alg, saveat = bgp.pde_setup.δt).u...)
        print("Solving PDEs: Step $pdeidx of $rand_pde.\u001b[1000D") # https://discourse.julialang.org/t/update-variable-in-logged-message-without-printing-a-new-line/32755
    end

    # Return
    return solns_all
end