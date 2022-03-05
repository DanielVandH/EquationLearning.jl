using EquationLearning
using Optim

"""
    set_parameters(n, dat, dat_idx, x_scale, t_scale)

Set the parameters used in the paper. Note that in this function we set ALL arguments, including 
the keyword arguments in functions that already have reasonable defaults. In practice you might only ever use
e.g. `bootstrap_gp(x, t, u, T, D′, R, α₀, β₀, γ₀, lowers, uppers)`.

# Arguments 
- `n`: This defines the study to use:
    - `n = 1`: A simulation study with no delay and a Fisher-Kolmogorov model.
    - `n = 2`: A simulation study with no delay and a Porous-Fisher model.
    - `n = 3`: A simulation study with no delay and diffusion given by `D₀ + D₁(u/K)²`.
    - `n = 4`: A simulation study with delay and a Fisher-Kolmogorov model.
    - `n = 5`: A simulation study with delay and a Porous-Fisher model.
    - `n = 6`: A simulation study with delay and diffusion given by `D₀ + D₁(u/K)²`.
    - `n = 7`: Data from Jin et al. (2016), assuming no delay and a Fisher-Kolmogorov model.
    - `n = 8`: Data from Jin et al. (2016), assuming no delay and a Porous-Fisher model.
    - `n = 9`: Data from Jin et al. (2016), assuming no delay and a diffusion function given by `D₀ + D₁(u/K)²`.
    - `n = 10`: Data from Jin et al. (2016), assuming delay and a Fisher-Kolmogorov model.
    - `n = 11`: Data from Jin et al. (2016), assuming delay and a Porous-Fisher model.
    - `n = 12`: Data from Jin et al. (2016), assuming delay and a diffusion function given by `D₀ + D₁(u/K)²`.
- `dat`: The data set to use from Jin et al. (2016).
- `dat_idx`: The data set from Jin et al. (2016) corresponding to `dat`.
    - `dat_idx = 1`: The 10,000 cells per well data.
    - `dat_idx = 2`: The 12,000 cells per well data.
    - `dat_idx = 3`: The 14,000 cells per well data.
    - `dat_idx = 4`: The 16,000 cells per well data.
    - `dat_idx = 5`: The 18,000 cells per well data.
    - `dat_idx = 6`: The 20,000 cells per well data.
- `x_scale`: The transformation used on the `x` data compared to the data presented by Jin et al. (2016).
- `t_scale`: The transformation used on the `t` data compared to the data presented by Jin et al. (2016).
"""
function set_parameters(n, dat, dat_idx, x_scale, t_scale)
    ## Initial data to use for generating synthetic data
    x₀ = dat.Position[dat.Time.==0.0]
    u₀ = dat.AvgDens[dat.Time.==0.0]
    ## Setup known functions and parameters
    lowers = Vector{Float64}([])
    uppers = Vector{Float64}([])
    if n ∈ 1:3 || n ∈ 7:9
        T = (t, α, p) -> 1.0
        T_params = nothing
        α₀ = Vector{Float64}([]) 
        push!(lowers, 0.5)
        push!(uppers, 2.0)
    elseif n ∈ 4:6 || n ∈ 10:12
        T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
        T_params = [1.0, 1.0]
        α₀ = [1.0, 1.0]
        push!(lowers, 0.5, 0.5)
        push!(uppers, 2.0, 2.0)
    end
    if n ∈ [1, 4, 7, 10]
        D = (u, β, p) -> β[1] * p[1]
        D′ = (u, β, p) -> 0.0
        D_params = 1.0
        β₀ = [1.0]
        push!(lowers, 0.5)
        push!(uppers, 2.0)
    elseif n ∈ [2, 5, 8, 11]
        D = (u, β, p) -> β[1] * p[2] * (u / p[1])
        D′ = (u, β, p) -> β[1] * p[2] / p[1]
        D_params = [1.7e-3 * x_scale^2, 1.0]
        β₀ = [1.0]
        push!(lowers, 0.5)
        push!(uppers, 2.0)
    elseif n ∈ [3, 6, 9, 12]
        D = (u, β, p) -> u > 0.0 ? β[1] * p[2] + β[2] * p[3] * (u / p[1])^2 : β[1] * p[2]
        D′ = (u, β, p) -> u > 0.0 ? β[2] * p[3] * 2 * u / p[1]^2 : 0.0
        D_params = [1.7e-3 * x_scale^2, 1.0, 1.0]
        β₀ = [1.0, 1.0]
        push!(lowers, 0.5, 0.5)
        push!(uppers, 2.0, 2.0)
    end
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    R_params = [1.7e-3 * x_scale^2, 1.0]
    γ₀ = [1.0]
    push!(lowers, 0.5)
    push!(uppers, 2.0)
    if n ∈ 1:6
        if n ∈ 1:3
            α = Vector{Float64}([])
        elseif n ∈ 4:6
            α₁ = [-1.0292, -3.3013, -3.1953, -2.9660, -1.2695, -4.0651][dat_idx]
            α₂ = [0.2110, 0.2293, 0.2761, 0.2180, 0.1509, 0.4166][dat_idx] * t_scale
            α = [α₁, α₂] ./ T_params
        end
        if n == 1 || n == 4
            β = [310.0, 250.0, 720.0, 570.0, 760.0, 1030.0][dat_idx] * t_scale / x_scale^2 / D_params
        elseif n == 2 || n == 5
            β = [1800.0, 1300.0, 3000.0, 2400.0, 2800.0, 2900.0][dat_idx] * t_scale / x_scale^2 / D_params[2]
        elseif n == 3 || n == 6
            β₁ = [95.7, 353.3, 482.1, 604.3, 804.0, 675.8][dat_idx] * t_scale / x_scale^2
            β₂ = [3987.1, 3166.4, 3775.0, 3773.8, 221.8, 1954.9][dat_idx] * t_scale / x_scale^2
            β = [β₁, β₂] ./ D_params[2:3]
        end
        if n ∈ 1:3
            γ = [0.044, 0.044, 0.048, 0.049, 0.054, 0.064][dat_idx] * t_scale / R_params[2]
        else
            γ = [0.0525, 0.0714, 0.0742, 0.0798, 0.0772, 0.0951][dat_idx] * t_scale / R_params[2]
        end
        ## Setup PDE for generating synthetic data
        δt = LinRange(extrema(dat.Time)..., 5)
        finalTime = maximum(dat.Time)
        N = 1000
        LHS = [0.0, 1.0, 0.0]
        RHS = [0.0, -1.0, 0.0]
        alg = nothing
        N_thin = 50
        num_restarts = 50
        x, t, u, _ = EquationLearning.generate_data(x₀, u₀, T, D, R, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
        x_pde = copy(x)
        t_pde = copy(t)
        u_pde = copy(u)
    else
        x = repeat(dat.Position, outer = 3)
        t = repeat(dat.Time, outer = 3)
        u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
        x_pde = dat.Position
        t_pde = dat.Time
        u_pde = dat.AvgDens
        δt = LinRange(extrema(dat.Time)..., 5)
        finalTime = maximum(dat.Time)
        LHS = [0.0, 1.0, 0.0]
        RHS = [0.0, -1.0, 0.0]
        alg = nothing
    end
    ## Setup the bootstrap
    bootₓ = LinRange(extrema(x)..., 80)
    bootₜ = LinRange(extrema(t)..., 75)
    B = 250
    τ = (0.0, 0.0)
    Optim_Restarts = 10
    constrained = false
    obj_scale_GLS = log
    obj_scale_PDE = log
    show_losses = false
    bootstrap_setup = EquationLearning.Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, show_losses)
    ## Setup the GP parameters 
    ℓₓ = log.([1e-4, 1.0])
    ℓₜ = log.([1e-4, 1.0])
    σ = log.([1e-1, 2std(u)])
    σₙ = log.([1e-5, 2std(u)])
    nugget = 1e-5
    GP_Restarts = 50
    gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
    gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)
    ## Setup the PDE parameters
    meshPoints = LinRange(extrema(x)..., 500)
    pde_setup = EquationLearning.PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)
    ## Other and return 
    optim_setup = Optim.Options(f_reltol = 1e-4, x_reltol = 1e-4, g_reltol = 1e-4, outer_f_reltol = 1e-4, outer_x_reltol = 1e-4, outer_g_reltol = 1e-4)
    if n ∈ 1:6
        return x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀,
        lowers, uppers, gp_setup, bootstrap_setup, optim_setup,
        pde_setup, D_params, R_params, T_params, α, β, γ
    else
        return x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀,
        lowers, uppers, gp_setup, bootstrap_setup, optim_setup,
        pde_setup, D_params, R_params, T_params
    end
end