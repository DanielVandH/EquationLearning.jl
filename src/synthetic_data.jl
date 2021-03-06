#####################################################################
## Script description: synthetic_data.jl
##
## This script defines one function, generate_data, for generating synthetic data.
##
#####################################################################

"""
    generate_data(x₀, u₀, T, D, R, α, β, γ, δt, finalTime; <keyword arguments>)

Generate synthetic data from values `(x₀, t₀)` at `t = 0` with exact mechanisms
`T`, `D`, and `R` for delay, diffusion, and reaction, respectively. 

# Arguments 
- `x₀`: The spatial mesh for points at `t = 0`.
- `u₀`: The density values at `t = 0` corresponding to the points in `x₀`.
- `T`: A delay function of the form `T(t, α, T_params)`.
- `D`: A diffusion function of the form `D(u, β, D_params)`.
- `R`: A reaction function of the form `R(u, γ, R_params)`.
- `D′`: The derivative of the diffusion function, given in the form `D′(u, β, D_params)`.
- `R′`: The reaction function, given in the form `R(u, γ, R_params)`.
- `α`: Exact values for the delay parameters.
- `β`: Exact values for the diffusion parameters.
- `γ`: Exact values for the reaction parameters. 
- `δt`: Times to save the solution to the differential equations at for generating the data.
- `finalTime`: The time to solve the differential equations up to.

# Keyword Arguments 
- `N = 1000`: The number of mesh points to use.
- `LHS = [0.0, 1.0, 0.0]`: Vector defining the left-hand boundary conditions for the PDE. See also the definitions of `(a₀, b₀, c₀)` in [`sysdegeneral!`](@ref).
- `RHS = [0.0, -1.0, 0.0]`: Vector defining the right-hand boundary conditions for the PDE. See also the definitions of `(a₁, b₁, c₁)` in [`sysdegeneral!`](@ref).
- `alg = nothing`: The algorithm to use for solving the differential equations.
- `N_thin = 100`: The number of points to take from the solution at each time.
- `num_restarts = 50`: The number of times to restart the optimiser for fitting the Gaussian process to the data.
- `D_params`: Additional known parameters for the diffusion function.
- `R_params`: Additional known parameters for the reaction function.
- `T_params`: Additional known parameters for the delay function.
"""
function generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime;
    N=1000, LHS=[0.0, 1.0, 0.0], RHS=[0.0, -1.0, 0.0],
    alg=nothing,
    N_thin=100, num_restarts=50, D_params, R_params, T_params)
    try
        D(u₀[1], β, D_params)
    catch
        throw("Either the provided vector of diffusion parameters, β₀ = $β, is not of adequate size, or D_params = $D_params has been incorrectly specified.")
    end
    try
        R(u₀[1], γ, R_params)
    catch
        throw("Either the provided vector of reaction parameters, γ₀ = $γ, is not of adequate size, or R_params = $R_params has been incorrectly specified.")
    end
    try
        T(0.0, α, T_params)
    catch
        throw("Either the provided vector of delay parameters, α₀ = $α, is not of adequate size, or T_params = $T_params has been incorrectly specified.")
    end
    # Generate the data
    meshPoints = LinRange(extrema(x₀)..., N)
    initialCondition = Dierckx.Spline1D(vec(x₀), vec(u₀); k=1)(meshPoints)

    # Define geometry
    a₀, b₀, c₀ = LHS
    a₁, b₁, c₁ = RHS
    h = diff(meshPoints)
    V = @views 1 / 2 * [h[1]; h[1:(N-2)] + h[2:(N-1)]; h[N-1]]

    # Define parameters
    Du = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10)) # We use dualcache so that we can easily integrate automatic differentiation into our ODE solver. The last argument is the chunk size, see the ForwardDiff docs for details. The /10 is just some heuristic I developed based on warnings given by PreallocationTools.
    D′u = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
    Ru = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
    R′u = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
    p = (N, V, h, a₀, b₀, c₀, a₁, b₁, c₁, Du, Ru, D′u, R′u, T, D, R, D′, R′, α, β, γ, D_params, R_params, T_params)

    # Solve 
    tspan = (0.0, finalTime)
    prob = ODEProblem(sysdegeneral!, initialCondition, tspan, p)
    sol = DifferentialEquations.solve(prob, CVODE_BDF(linear_solver=:Band, jac_upper=1, jac_lower=1); saveat=δt)
    u = abs.(hcat(sol.u...))
    # Add noise 
    u .+= abs.(u) .^ (0.45) .* randn(size(u))
    u .= max.(u, 0.0)
    # Thin the data 
    thin_idx = 1:trunc(Int64, N / N_thin):N
    if thin_idx[end] ≠ N
        thin_idx = [thin_idx; N]
    end
    x = repeat(meshPoints[thin_idx], outer=length(sol.t))
    t = repeat(sol.t, inner=length(thin_idx))
    u = vec(u[thin_idx, :])
    return x, t, u, []
end