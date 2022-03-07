#####################################################################
## Script description: synthetic_data.jl
##
## This script defines one function, generate_data, for generating synthetic data.
##
#####################################################################

"""
    generate_data(x₀, u₀, T, D, R, α, β, γ, δt, finalTime; <keyword arguments>)

Generate synthetic data from values `(x₀, t₀)` at `t = 0` with exact mechanisms
`T`, `D`, and `R` for delay, diffusion, and reaction, respectively. A Gaussian process 
is fit to the data to smooth it, and then that curve is used to generate a smooth initial 
condition which is then perturbed by noise according to the estimated signal noise from the 
fitted Gaussian process.

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
    N = 1000, LHS = [0.0, 1.0, 0.0], RHS = [0.0, -1.0, 0.0], 
    alg = nothing, 
    N_thin = 100, num_restarts = 50, D_params, R_params, T_params)
    @assert length(x₀) == length(u₀) "The lengths of the provided data vectors must all be equal."
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

    # Fit a smooth curve using GPs 
    meanFunc = MeanZero()
    covFunc = SE(2.0, 2.0)
    x₀ = Matrix(vec(x₀)')
    u₀ = vec(u₀)
    x_min = minimum(x₀)
    x_rng = maximum(x₀) - x_min
    X = (x₀ .- x_min) / x_rng
    gp = GPE(X, u₀, meanFunc, covFunc, 2.0)
    plan, _ = LHCoptim(num_restarts, length(GaussianProcesses.get_params(gp)), 2000)
    new_params = scaleLHC(plan, [(log(1e-4), log(1.0)), (log(1e-5), log(2std(u₀))), (log(1e-5), log(2std(u₀)) + 1e-5)])'
    obj_values = zeros(num_restarts)
    for j = 1:num_restarts
        try
            @views GaussianProcesses.set_params!(gp, new_params[:, j])
            GaussianProcesses.optimize!(gp)
            obj_values[j] = gp.target
        catch err
            println(err)
            obj_values[j] = -Inf
        end
    end
    opt_model = findmax(obj_values)[2]
    @views GaussianProcesses.set_params!(gp, new_params[:, opt_model])
    GaussianProcesses.optimize!(gp)

    # Generate the data
    meshPoints = LinRange(extrema(x₀)..., N)
    initialCondition, _ = predict_f(gp, (vec(meshPoints) .- x_min) / x_rng)
    initialCondition .= max.(initialCondition, 0.0)

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
    ode_fnc = ODEFunction(sysdegeneral!)
    prob = ODEProblem(ode_fnc, initialCondition, tspan, p, jac_prototype = Tridiagonal(zeros(N, N)))
    sol = DifferentialEquations.solve(prob, alg; saveat = δt)
    u = abs.(hcat(sol.u...))
    # Add noise 
    u .+= exp(gp.logNoise.value) .* randn(size(u))
    u .= max.(u, 0.0)
    # Thin the data 
    thin_idx = 1:trunc(Int64, N / N_thin):N
    if thin_idx[end] ≠ N
        thin_idx = [thin_idx; N]
    end
    x = repeat(meshPoints[thin_idx], outer = length(sol.t))
    t = repeat(sol.t, inner = length(thin_idx))
    u = vec(u[thin_idx, :])
    return x, t, u, gp
end