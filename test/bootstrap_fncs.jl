@testset "Bootstrapping and PDEs" begin
    LinearAlgebra.BLAS.set_num_threads(1)
    x_scale = 1000.0
    t_scale = 24.0
    alphabet = join('a':'z')
    fontsize = 20
    colors = [:black, :blue, :red, :magenta, :green]
    legendentries = OrderedDict("0" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[1]),
        "12" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[2]),
        "24" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[3]),
        "36" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[4]),
        "48" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[5]))
    K = 1.7e-3 * x_scale^2
    δt = LinRange(0.0, 48.0 / t_scale, 5)
    finalTime = 48.0 / t_scale
    N = 1000
    LHS = [0.0, 1.0, 0.0]
    RHS = [0.0, -1.0, 0.0]
    alg = Tsit5()
    N_thin = 38
    meshPoints = LinRange(25.0 / x_scale, 1875.0 / x_scale, 500)
    pde_setup = PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)
    nₓ = 30
    nₜ = 30
    bootₓ = LinRange(25.0 / x_scale, 1875.0 / x_scale, nₓ)
    bootₜ = LinRange(0.0, 48.0 / t_scale, nₜ)
    B = 100
    τ = (0.0, 0.0)
    Optim_Restarts = 10
    constrained = false
    obj_scale_GLS = log
    obj_scale_PDE = log
    show_losses = false
    init_weight = 10.0
    bootstrap_setup = Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, init_weight, show_losses)
    num_restarts = 250
    ℓₓ = log.([1e-7, 2.0])
    ℓₜ = log.([1e-7, 2.0])
    nugget = 1e-5
    GP_Restarts = 250
    optim_setup = Optim.Options(iterations=10, f_reltol=1e-4, x_reltol=1e-4, g_reltol=1e-4, outer_f_reltol=1e-4, outer_x_reltol=1e-4, outer_g_reltol=1e-4)

    # Is generate_data working?
    Random.seed!(51021)
    x₀ = [0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.025, 1.075, 1.125, 1.175, 1.225, 1.275, 1.325, 1.375, 1.425, 1.475, 1.525, 1.575, 1.625, 1.675, 1.725, 1.775, 1.825, 1.875]
    u₀ = [312.0, 261.0, 233.0, 303.0, 252.0, 252.0, 228.0, 242.0, 238.0, 191.0, 135.0, 140.0, 131.0, 79.3, 46.6, 37.3, 46.6, 46.6, 74.6, 65.3, 28.0, 37.3, 14.0, 4.66, 14.0, 0.0, 9.32, 0.0, 28.0, 97.89999999999999, 172.0, 252.0, 368.0, 350.0, 410.0, 331.0, 350.0]
    T = (t, α, p) -> 1.0
    D = (u, β, p) -> β[1] * p[1]
    D′ = (u, β, p) -> 0.0
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    α = Vector{Float64}([])
    β = [301.0] * t_scale / x_scale^2
    γ = [0.044] * t_scale
    T_params = Vector{Float64}([])
    D_params = [1.0]
    R_params = [K, 1.0]
    x, t, u, datgp = generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
    x_pde = copy(x)
    t_pde = copy(t)
    u_pde = copy(u)
    @test all((mean(x), median(x), sum(x), median(x .^ 2)) .≈ ((0.9878378378378379, 0.9885135135135136, 197.56756756756758, 0.9777076231637043)))
    @test all((mean(t), median(t), sum(t), median(t .^ 2), median(t .* x)) .≈ ((1.0, 1.0, 200.0, 1.0, 0.7238175675675675)))
    @test all((mean(u), median(u), median(x .+ t .+ u), sum(u), mean(u .* x), median(u .^ 2 + t .^ 2)) .≈ ((392.6780194026992, 324.3555732722282, 325.90197272833564, 78535.60388053984, 378.7252040143025, 105208.28579627146)))

    # Is precompute_gp_mean working correctly?
    Random.seed!(51565021)
    σ = log.([1e-6, 7std(u)])
    σₙ = log.([1e-6, 7std(u)])
    gp, μ, L = precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, 250, bootstrap_setup)
    gp_setup = GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts=250, μ, L, nugget, gp)
    @test gp.mll ≈ -916.4090604839076
    @test gp.kernel.iℓ2 ≈ [24.84827104335778, 0.503514925015118]
    @test gp.kernel.σ2 ≈ 470228.28092725377
    @test gp.logNoise.value ≈ 2.76897258470674
    @test sum(μ) ≈ 118447.98230142379 atol = 1e-2
    @test prod(abs.(log.(abs.(μ))) .^ (0.0001)) ≈ 1.972569094606653 atol = 1e-2
    @test median(μ) ≈ 328.7270614832669 atol = 1e-2
    @test mean(μ) ≈ 32.902217305951055 atol = 1e-2
    @test mean(sin.(μ)) ≈ 0.0032722144858676003 atol = 1e-2
    @test mean(L) ≈ 0.050455323132358405 atol = 1e-2
    @test sum(Symmetric(L)) ≈ 1630.943707819713 atol = 1e-2
    @test Symmetric(L)[2, 2] ≈ 6.408267809805805 atol = 1e-2
    @test sum(eigen(Symmetric(L)[1:200, 1:200]).values) ≈ 39.37032898512207 atol = 1e-2
    @test median(Symmetric(L)) ≈ 0.0 atol = 1e-2

    # Is bootstrap_gp working correctly?
    Random.seed!(99992001)
    T = (t, α, p) -> 1.0
    D = (u, β, p) -> β[1] * p[1]
    D′ = (u, β, p) -> 0.0
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = Vector{Float64}([])
    D_params = [301.0] * t_scale / x_scale^2
    R_params = [K, 0.044 * t_scale]
    α₀ = Vector{Float64}([])
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.9]
    uppers = [0.99, 1.1]
    bootstrap_setup = @set bootstrap_setup.B = 20
    bootstrap_setup = @set bootstrap_setup.show_losses = false
    bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 2
    bgp1 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
    @test size(bgp1.delayBases) == (0, 20)
    @test bgp1.diffusionBases ≈ Matrix([0.973216 0.878354 0.886621 0.972151 0.919623 0.90151 0.823659 0.895349 0.793193 0.933561 0.85901 0.86767 0.850462 0.743167 0.946262 0.896715 0.852373 0.967358 0.99301 0.946777]) atol = 1e-5
    @test bgp1.reactionBases ≈ Matrix([1.00429 1.00969 0.987766 1.00221 1.00874 0.99997 0.99786 1.00827 0.993301 0.996773 1.00922 1.00111 0.997528 0.992874 1.00835 1.00381 0.996453 0.995867 1.01418 0.999509]) atol = 1e-5
    @test bgp1.gp === gp
    @test mean(bgp1.zvals) ≈ -0.0025212240296433952
    @test bgp1.zvals[2000, 7] ≈ -0.3054212070111045
    @test bgp1.Xₛ[1:2, 1:5] ≈ [-0.0277778 0.00766284 0.0431034 0.0785441 0.113985
        0.0 0.0 0.0 0.0 0.0] atol = 1e-6
    @test bgp1.Xₛⁿ[1:2, 400:407] ≈ [0.599138 0.662931 0.726724 0.790517 0.85431 0.918103 0.981897 1.04569
        0.896552 0.896552 0.896552 0.896552 0.896552 0.896552 0.896552 0.896552] atol = 1e-5
    @test bgp1.bootₓ ≈ collect(bootₓ)
    @test bgp1.bootₜ ≈ collect(bootₜ)
    @test bgp1.T === T
    @test bgp1.D === D
    @test bgp1.D′ === D′
    @test bgp1.R === R
    @test bgp1.R′ === R′
    @test bgp1.D_params == Matrix(repeat(D_params, 20)')
    @test bgp1.R_params ≈ [K K K K K K K K K K K K K K K K K K K K; 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale 0.044*t_scale]
    @test bgp1.μ ≈ μ
    @test bgp1.L ≈ L
    @test bgp1.gp_setup === gp_setup
    @test bgp1.pde_setup === pde_setup
    @test bgp1.bootstrap_setup === bootstrap_setup

    # Is density_values working correctly?
    trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp1; level=0.05, diffusion_scales=D_params[1] .* x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
    @test size(trv) == (0, 20)
    @test dr ≈ bgp1.diffusionBases * D_params[1] * x_scale^2 / t_scale
    @test rr ≈ bgp1.reactionBases * R_params[2] / t_scale
    @test d == 1
    @test r == 1
    @test size(delayCIs) == (0, 2)
    @test diffusionCIs ≈ [230.846 296.066] atol = 1e-3
    @test reactionCIs ≈ [0.0435685 0.0445301] atol = 1e-6

    # Is curve_values working correctly?
    Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp1; level=0.05, x_scale=x_scale, t_scale=t_scale)
    @test Tu_vals[1] == Tu_vals[2] == Tu_vals[3] == repeat([1.0], 500)
    @test unique(Du_vals[1]) ≈ [269.39564107403555] atol = 1e-2
    @test unique(Du_vals[2]) ≈ [230.84585165548006] atol = 1e-2
    @test unique(Du_vals[3]) ≈ [296.06600312482414] atol = 1e-2
    @test Ru_vals[1][[1, 10, 59, 100, 250, 397, 500]] ≈ vec([0.10215351577478515
        1.0035180776273855
        5.518738862966758
        8.787544520187224
        16.79276363772164
        18.612744410671173
        16.334355814912406]) atol = 1e-4
    @test Ru_vals[2][[4, 100, 19, 500, 325]] ≈ [0.4005631418841285
        8.689291672259712
        1.8614775520470275
        16.151722660202392
        18.26007595965338] atol = 1e-2
    @test Ru_vals[3][[1, 2, 55, 100, 100, 325]] ≈ [0.10324078282440266
        0.205574146665954
        5.230070013413127
        8.88107441518282
        8.88107441518282
        18.66309700965525] atol = 1e-2
    @test u_vals[[1, 3, 5, 100, 500]] ≈ [2.3216216643979295
        6.9366378754184215
        11.551654086438912
        230.76492410991227
        1153.7681663140106]
    @test t_vals[[1, 2, 3, 4, 100, 250, 377]] ≈ [0.0
        0.004008016032064128
        0.008016032064128256
        0.012024048096192385
        0.3967935871743487
        0.9979959919839679
        1.5070140280561122]

    # Is compute_initial_conditions working correctly?
    nodes, weights = gausslegendre(5)
    tr = bgp1.delayBases
    dr = bgp1.diffusionBases
    rr = bgp1.reactionBases
    B = size(dr, 2)
    prop_samples = 1.0
    rand_pde = convert(Int64, trunc(prop_samples * B))
    N = length(bgp1.pde_setup.meshPoints)
    M = length(bgp1.pde_setup.δt)
    solns_all = zeros(N, rand_pde, M)
    ICType = "data"
    initialCondition_all = EquationLearning.compute_initial_conditions(x_pde, t_pde, u_pde, bgp1, ICType)
    @test size(initialCondition_all) == (N, B)
    @test initialCondition_all[:, 1] == initialCondition_all[:, 10] == initialCondition_all[:, 20]
    @test initialCondition_all[1:20, 5] ≈ [301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        301.84804479333087,
        299.6011638934639,
        295.2256589831967,
        290.8501540729294,
        286.4746491626621,
        282.09914425239486,
        277.7236393421276] atol = 1e-2
    @test initialCondition_all[[19, 59, 300, 301, 391, 500], 1] ≈ [282.09914425239486,
        261.4782754485342,
        26.724265114140444,
        25.81300127559665,
        30.133934991179117,
        366.5262850744492] atol = 1e-2
    @test initialCondition_all ≈ EquationLearning.compute_initial_conditions(x_pde, t_pde, u_pde, ICType, bgp1, N, B, meshPoints)
    ICType = "gp"
    initialCondition_all = EquationLearning.compute_initial_conditions(x_pde, t_pde, u_pde, bgp1, ICType)
    @test size(initialCondition_all) == (N, B)
    @test initialCondition_all[391, 4] ≈ 54.136239631410625 atol = 1e-2
    @test initialCondition_all[[1, 2, 5, 15, 100, 99, 500, 399, 4], 1] ≈ vec([304.66050973379447
        302.53289434633825
        296.15004818396966
        274.8738943094077
        238.90677330388684
        239.10847806836728
        337.1533730159008
        83.1298542579187
        298.2776635714259]) atol = 1e-2
    @test initialCondition_all[[1, 100, 400, 499, 2, 3, 50], 7] ≈ vec([303.3455508021752
        232.67652427131264
        88.91228170700387
        362.6555237115071
        301.76756285136713
        300.189574900559
        252.31154742255802]) atol = 1e-2
    @test initialCondition_all ≈ EquationLearning.compute_initial_conditions(x_pde, t_pde, u_pde, ICType, bgp1, N, B, meshPoints)

    # Is compute_valid_pde_indices working correctly?
    num_u = 500
    num_t = 500
    idx = EquationLearning.compute_valid_pde_indices(u_pde, num_t, num_u, nodes, weights, bgp1)
    u_vals = range(minimum(bgp1.gp.y), maximum(bgp1.gp.y), length=num_u)
    t_vals = collect(range(minimum(bgp1.Xₛⁿ[2, :]), maximum(bgp1.Xₛⁿ[2, :]), length=num_t))
    Tuv = zeros(num_t, 1)
    Duv = zeros(num_u, 1)
    max_u = maximum(u_pde)
    for j in idx
        Duv .= bgp1.D.(u_vals, Ref(bgp1.diffusionBases[:, j]), Ref(bgp1.D_params))
        Tuv .= bgp1.T.(t_vals, Ref(bgp1.delayBases[:, j]), Ref(bgp1.T_params))
        Reaction = u -> bgp1.R(max_u / 2 * (u + 1), bgp1.reactionBases[:, j], bgp1.R_params) # missing a max_u/2 factor in front for this new integral, but thats fine since it doesn't change the sign
        Ival = dot(weights, Reaction.(nodes))
        @test !(any(Duv .< 0) || any(Tuv .< 0) || Ival < 0)
    end
    @test idx == collect(1:20)
    @test EquationLearning.compute_valid_pde_indices(u_pde, num_t, num_u, nodes, weights, bgp1) == EquationLearning.compute_valid_pde_indices(bgp1, u_pde, num_t, num_u, bgp1.bootstrap_setup.B, bgp1.delayBases, bgp1.diffusionBases, bgp1.reactionBases, nodes, weights, bgp1.D_params, bgp1.R_params, bgp1.T_params)

    # Is boot_pde_solve working correctly?
    pde_gp1 = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="gp")
    pde_data1 = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="data")
    @test pde_gp1[1:4, 1:2, 1] ≈ [304.661 283.683
        302.533 282.895
        300.405 282.107
        298.278 281.319] atol = 1e-2
    @test pde_gp1[97:103, 10:13, 2] ≈ [364.326 356.213 364.092 368.716
        363.274 355.299 363.323 367.849
        362.187 354.354 362.52 366.948
        361.068 353.378 361.67 365.998
        359.912 352.367 360.783 365.014
        358.723 351.322 359.849 363.98
        357.496 350.241 358.875 362.908] atol = 1e-2
    @test pde_gp1[393:399, 17, 3:4] ≈ [238.849 388.418
        245.628 396.517
        252.435 404.767
        259.373 412.971
        266.325 421.325
        273.404 429.613
        280.482 438.05] atol = 1e-2
    @test pde_gp1[192:195, 17:20, [5, 1]] ≈ [525.701 566.008 525.92 570.77
        520.439 560.752 520.749 566.356
        515.22 555.517 515.62 561.977
        510.052 550.311 510.552 557.635;;;
        64.7322 75.9872 54.2694 74.6537
        63.1411 74.2223 52.8052 73.2691
        61.55 72.4573 51.341 71.8846
        59.959 70.6924 49.8767 70.5] atol = 1e-2
    @test pde_data1[393:399, 10:13, 1] ≈ [35.4939 35.4939 35.4939 35.4939
        39.0778 39.0778 39.0778 39.0778
        43.8411 43.8411 43.8411 43.8411
        48.6043 48.6043 48.6043 48.6043
        53.3675 53.3675 53.3675 53.3675
        58.1307 58.1307 58.1307 58.1307
        62.8939 62.8939 62.8939 62.8939] atol = 1e-3
    @test pde_data1[97:103, 10:13, 2] ≈ [357.624 359.838 358.574 358.073
        356.69 358.811 357.585 357.149
        355.599 357.848 356.584 356.096
        354.627 356.79 355.563 355.14
        353.497 355.792 354.527 354.055
        352.483 354.696 353.467 353.061
        351.308 353.657 352.39 351.936] atol = 1e-2
    @test pde_data1[393:399, 17, 3:4] ≈ [210.908 356.776
        217.975 365.303
        224.913 374.032
        232.217 382.705
        239.371 391.578
        246.894 400.376
        254.243 409.371] atol = 1e-2
    @test pde_data1[2:10, 17:20, [5, 1]] ≈ [1024.55 1021.54 1036.74 1025.15
        1024.47 1021.46 1036.59 1024.99
        1024.46 1021.46 1036.65 1025.07
        1024.34 1021.34 1036.47 1024.87
        1024.29 1021.3 1036.49 1024.9
        1024.12 1021.13 1036.27 1024.66
        1024.03 1021.05 1036.25 1024.66
        1023.81 1020.85 1035.99 1024.37
        1023.68 1020.73 1035.93 1024.33;;;
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848] atol = 1e-1

    # Is compute_ribbon_features working correctly?
    X = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20]
    x_mean = [mean([1 2 3 4 5]), mean([6 7 8 9 10]), mean([11 12 13 14 15]), mean([16 17 18 19 20])]
    x_lower = [quantile([1, 2, 3, 4, 5], 0.025), quantile([6, 7, 8, 9, 10], 0.025), quantile([11, 12, 13, 14, 15], 0.025), quantile([16, 17, 18, 19, 20], 0.025)]
    x_upper = [quantile([1, 2, 3, 4, 5], 0.975), quantile([6, 7, 8, 9, 10], 0.975), quantile([11, 12, 13, 14, 15], 0.975), quantile([16, 17, 18, 19, 20], 0.975)]
    @test (x_mean, x_lower, x_upper) == EquationLearning.compute_ribbon_features(X)

    # Is pde_values working correctly?
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data1, bgp1)
    for j in 1:length(bgp1.pde_setup.δt)
        @test all((soln_vals_mean[:, j], soln_vals_lower[:, j], soln_vals_upper[:, j]) .≈ EquationLearning.compute_ribbon_features(pde_data1[:, :, j]))
    end

    # Is delay_product working correctly?
    @test all(Du_vals .≈ delay_product(bgp1, 0.5; x_scale, t_scale))
    @test all(Ru_vals .≈ delay_product(bgp1, 0.5; x_scale, t_scale, type="reaction"))

    # Is error_comp working correctly?
    err_CI1 = error_comp(bgp1, pde_data1, x_pde, t_pde, u_pde)
    err_CI2 = error_comp(bgp1, pde_data1, x_pde, t_pde, u_pde; compute_mean=true)
    @test err_CI1 ≈ [2.006713037478207, 2.482940618305627] atol = 1e-2
    @test err_CI2 ≈ 2.1809270509678784 atol = 1e-2

    # Is AIC working correctly?
    AICs1 = AIC(bgp1, x_pde, t_pde, u_pde; correct=false)
    AICs2 = AIC(bgp1, x_pde, t_pde, u_pde; correct=true)
    AICs3 = AIC(bgp1, x_pde, t_pde, u_pde; correct=false, pde_solns=pde_data1)
    AICs4 = AIC(bgp1, x_pde, t_pde, u_pde; correct=false, pde_solns=pde_gp1)
    AICs5 = AIC(bgp1, x_pde, t_pde, u_pde; correct=true, pde_solns=pde_data1)
    AICs6 = AIC(bgp1, x_pde, t_pde, u_pde; correct=true, pde_solns=pde_gp1)
    @test AICs1 == AICs4
    @test AICs2 == AICs6
    @test AICs2 ≈ AICs1 .+ 2 * 12 / (200 - 2)
    @test AICs3 ≈ vec([1043.905766631577
        1071.6896523941698
        1148.092204452611
        1051.419028965671
        1050.65488715628
        1081.727391124466
        1133.859620820133
        1062.9999925697714
        1170.2188769940603
        1084.480809147456
        1084.2213050270266
        1095.404979151756
        1118.1077257017516
        1206.4768500848252
        1041.53040020126
        1070.976793921534
        1121.1795747599606
        1081.476714176071
        1032.0366860231716
        1067.5249800589413]) atol = 1e-2
    @test AICs5 ≈ AICs3 .+ 2 * 12 / (200 - 2)

    # Is classify_Δᵢ working correctly? 
    @test EquationLearning.classify_Δᵢ(0.5) == 1
    @test EquationLearning.classify_Δᵢ(3.1) == 2
    @test EquationLearning.classify_Δᵢ(8.01) == 3
    @test EquationLearning.classify_Δᵢ.([0.5, 0.01, 3.01, 3.1, 7.9, 8.01, 15.0]) == [1, 1, 2, 2, 2, 3, 3]

    # Is compare_AICs working correctly?
    @test compare_AICs(12.0, 12.1, 38.0) == (1, 1, 3)
    @test compare_AICs(12.0, 12.1, 15.0, 16.0, 40.0) === (1, 1, 1, 2, 3)
    @test compare_AICs(12.0, 12.0) == (1, 1)
    AICs = [12.0 14.1 20.0; 20.7 14.3 10.0; 10.0 9.0 27.0; 19.0 7.0 13.0]
    class1 = compare_AICs(AICs[1, :]...)
    class2 = compare_AICs(AICs[2, :]...)
    class3 = compare_AICs(AICs[3, :]...)
    class4 = compare_AICs(AICs[4, :]...)
    results = [class1; class2; class3; class4]
    AIC_props = [2/4 0/4 2/4; 3/4 1/4 0/4; 1/4 2/4 1/4]
    @test AIC_props ≈ compare_AICs(AICs[:, 1], AICs[:, 2], AICs[:, 3])
    @test AIC_props ≈ compare_AICs([AICs[:, 1]; 1], AICs[:, 2], AICs[:, 3])

    # Can we fit a delay model?
    Random.seed!(5106221)
    bootstrap_setup = @set bootstrap_setup.B = 15
    bootstrap_setup = @set bootstrap_setup.show_losses = false
    bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
    x₀ = [0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.025, 1.075, 1.125, 1.175, 1.225, 1.275, 1.325, 1.375, 1.425, 1.475, 1.525, 1.575, 1.625, 1.675, 1.725, 1.775, 1.825, 1.875]
    u₀ = [312.0, 261.0, 233.0, 303.0, 252.0, 252.0, 228.0, 242.0, 238.0, 191.0, 135.0, 140.0, 131.0, 79.3, 46.6, 37.3, 46.6, 46.6, 74.6, 65.3, 28.0, 37.3, 14.0, 4.66, 14.0, 0.0, 9.32, 0.0, 28.0, 97.89999999999999, 172.0, 252.0, 368.0, 350.0, 410.0, 331.0, 350.0]
    T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
    D = (u, β, p) -> β[1] * p[1]
    D′ = (u, β, p) -> 0.0
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    α = [-1.50, 0.31 * t_scale]
    β = [571.0] * t_scale / x_scale^2
    γ = [0.081] * t_scale
    T_params = [1.0, 1.0]
    D_params = [1.0]
    R_params = [K, 1.0]
    x, t, u, datgp = EquationLearning.generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
    x_pde = copy(x)
    t_pde = copy(t)
    u_pde = copy(u)
    σ = log.([1e-6, 7std(u)])
    σₙ = log.([1e-6, 7std(u)])
    gp, μ, L = precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, 250, bootstrap_setup)
    gp_setup = GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts=250, μ, L, nugget, gp)
    Random.seed!(510226345431)
    T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
    D = (u, β, p) -> β[1] * p[1]
    D′ = (u, β, p) -> 0.0
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = [-1.5, 0.431 * t_scale]
    D_params = [571.0 * t_scale / x_scale^2]
    R_params = [K, 0.081 * t_scale]
    α₀ = [1.0, 1.0]
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99, 0.99, 0.99]
    uppers = [1.01, 1.01, 1.01, 1.01]
    bgp2 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
    pde_gp2 = boot_pde_solve(bgp2, x_pde, t_pde, u_pde; ICType="gp")
    @test bgp2.delayBases ≈ [0.744141 0.82586 0.785523 0.75064 0.918876 0.906362 0.96798 0.860483 0.775821 0.566292 0.846334 0.670889 0.797703 0.541827 1.03082
        0.618348 0.539515 0.537072 0.62087 0.639958 0.63816 0.671787 0.585295 0.631955 0.49497 0.597889 0.489293 0.622956 0.519537 0.682896] atol = 1e-2
    @test bgp2.diffusionBases ≈ [1.04983 0.91423 1.02901 0.976896 0.874503 0.90945 0.894913 0.879442 0.906306 0.894008 0.93522 0.981373 0.968472 0.94706 0.928471] atol = 1e-2
    @test bgp2.reactionBases ≈ [1.00849 1.01644 1.00108 0.999999 0.996091 0.999452 0.996499 1.00401 0.984982 0.995392 1.00142 1.00949 0.995022 0.999892 1.00255] atol = 1e-2
    @test bgp2.gp === gp
    @test bgp2.Xₛ[1:2, [1, 2, 40, 599, 600, 900]] ≈ [-0.0277778 0.00766284 0.291188 0.964559 1.0 1.0
        0.0 0.0 0.0344828 0.655172 0.655172 1.0] atol = 1e-3
    @test bgp2.Xₛⁿ[1:2, [1, 2, 40, 599, 600, 900]] ≈ [0.025 0.0887931 0.599138 1.81121 1.875 1.875
        0.0 0.0 0.0689655 1.31034 1.31034 2.0] atol = 1e-3
    @test bgp2.Xₛⁿ ≈ bgp2.Xₛ .* [maximum(x) - minimum(x), maximum(t) - minimum(t)] .+ [minimum(x), minimum(t)] atol = 1e-3
    @test bgp2.bootₓ ≈ collect(bgp2.bootₓ) atol = 1e-3
    @test bgp2.bootₜ ≈ collect(bgp2.bootₜ) atol = 1e-3
    @test bgp2.T === T
    @test bgp2.D === D
    @test bgp2.R === R
    @test bgp2.D′ === D′
    @test bgp2.R′ === R′
    @test bgp2.D_params ≈ [0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704 0.013704] atol = 1e-4
    @test bgp2.R_params ≈ [K K K K K K K K K K K K K K K; 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944 1.944] atol = 1e-3
    @test bgp2.T_params ≈ [-1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5; 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344 10.344] atol = 1e-3
    @test bgp2.μ === μ
    @test bgp2.L === L
    @test bgp2.gp_setup === gp_setup
    @test bgp2.bootstrap_setup === bootstrap_setup
    @test bgp2.pde_setup === pde_setup
    trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp2; level=0.05, delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=D_params[1] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
    @test trv ≈ [-1.11621 -1.23879 -1.17828 -1.12596 -1.37831 -1.35954 -1.45197 -1.29072 -1.16373 -0.849438 -1.2695 -1.00633 -1.19655 -0.81274 -1.54623
        0.266508 0.232531 0.231478 0.267595 0.275822 0.275047 0.28954 0.252262 0.272372 0.213332 0.25769 0.210885 0.268494 0.223921 0.294328] atol = 1e-1
    @test dr ≈ [599.456 522.026 587.567 557.808 499.341 519.296 510.995 502.161 517.501 510.478 534.011 560.364 552.998 540.771 530.157] atol = 1e-1
    @test rr ≈ [0.0816874 0.0823317 0.0810878 0.0809999 0.0806834 0.0809556 0.0807164 0.0813246 0.0797836 0.0806268 0.0811147 0.0817689 0.0805968 0.0809912 0.0812066] atol = 1e-1
    @test tt == 2
    @test d == 1
    @test r == 1
    @test delayCIs ≈ [-1.51324 -0.825584
        0.211741 0.292652] atol = 1e-1
    @test diffusionCIs ≈ [500.328 595.295] atol = 1e-1
    @test reactionCIs ≈ [0.0800682 0.0821347] atol = 1e-1
    Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp2; level=0.05, x_scale=x_scale, t_scale=t_scale)
    @test Tu_vals[1][1:10] ≈ vec([0.2334841022502253
        0.23784249679123065
        0.24225802547018063
        0.24673049317907828
        0.2512596642730799
        0.25584526170347277
        0.26048696619178724
        0.2651844154486411
        0.2699372034409313
        0.2747448797109892]) atol = 1e-1
    @test Tu_vals[2][1:10] ≈ vec([0.18055511307460306
        0.18475545986088773
        0.18903098816595704
        0.19338199724375932
        0.19780873622842626
        0.20231140203725612
        0.20689013729255187
        0.21154502826917498
        0.216276102874953
        0.2210833286713381]) atol = 1e-1
    @test Tu_vals[3][1:10] ≈ vec([0.30459222959515697
        0.3090980091366651
        0.3136403633540367
        0.3182187060392382
        0.3228324253520182
        0.32748088393591795
        0.33216341906968677
        0.33687934285458876
        0.34162794243800926
        0.34640848027369]) atol = 1e-1
    @test Du_vals[1][1:10] ≈ vec([536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551]) atol = 1e-1
    @test Du_vals[2][1:10] ≈ vec([500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491]) atol = 1e-1
    @test Du_vals[3][1:10] ≈ vec([595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887]) atol = 1e-1
    @test Ru_vals[1][1:10] ≈ vec([0.24335096986060162
        0.4793280870828808
        0.7144884269946098
        0.948831989595788
        1.1823587748864157
        1.4150687828664923
        1.6469620135360188
        1.8780384668949957
        2.1082981429434207
        2.337741041681296]) atol = 1e-1
    @test Ru_vals[2][1:10] ≈ vec([0.2403783161421765
        0.47347285494130054
        0.7057605937729302
        0.9372415326370656
        1.1679156715337062
        1.3977830104628526
        1.6268435494245046
        1.8550972884186623
        2.082544227445325
        2.309184366504494]) atol = 1e-1
    @test Ru_vals[3][1:10] ≈ vec([0.24658244051776848
        0.4856931106102229
        0.7239761573621563
        0.9614315807735685
        1.1980593808444593
        1.4338595575748292
        1.668832110964678
        1.9029770410140059
        2.1362943477228122
        2.368784031091098]) atol = 1e-1
    @test u_vals[492:500] ≈ vec([1439.9648553014622
        1442.8914487124116
        1445.818042123361
        1448.7446355343106
        1451.67122894526
        1454.5978223562095
        1457.5244157671589
        1460.4510091781083
        1463.3776025890577]) atol = 1e-1
    @test t_vals[39:43] ≈ vec([0.1523046092184369
        0.156312625250501
        0.16032064128256512
        0.16432865731462926
        0.1683366733466934]) atol = 1e-1
    Du_vals = delay_product(bgp2, 0.5; x_scale, t_scale)
    @test Du_vals[1][1:10] ≈ vec([463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516
        463.6130610175516]) atol = 1e-1
    @test Du_vals[2][1:10] ≈ vec([428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782]) atol = 1e-1
    @test Du_vals[3][1:10] ≈ vec([520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358]) atol = 1e-1
    Ru_vals = delay_product(bgp2, 0.5; x_scale, t_scale, type="reaction")
    @test Ru_vals[1][39:48] ≈ vec([7.464926608887905
        7.642069740428939
        7.818506876129778
        7.994238015990422
        8.169263160010875
        8.343582308191131
        8.517195460531195
        8.690102617031066
        8.862303777690743
        9.03379894251022]) atol = 1e-1
    @test Ru_vals[2][79:88] ≈ vec([13.4388611407641
        13.581796447996014
        13.724054055447443
        13.865633963118391
        14.006536171008852
        14.146760679118835
        14.286307487448333
        14.425176595997346
        14.563368004765879
        14.700881713753926]) atol = 1e-1
    @test Ru_vals[3][191:200] ≈ vec([27.17482229897456
        27.24700797666704
        27.318463860561167
        27.389189950656935
        27.459186246954346
        27.52845274945341
        27.596989458154116
        27.664796373056465
        27.731873494160457
        27.798220821466092]) atol = 1e-1
    Random.seed!(29991)
    res = compare_AICs(x_pde, t_pde, u_pde, bgp1, bgp2)
    @test res ≈ [0.0 0.0 1.0; 1.0 0.0 0.0]
    @test pde_gp2[[1:10..., 73, 101], [1, 2, 4, 5, 10, 11, 13, 15], [1, 2, 3, 4, 5]] ≈ [
        333.873 350.709 335.687 307.235 303.404 394.391 320.143 377.138
        331.823 349.02 334.079 306.231 302.447 389.805 318.476 373.977
        329.773 347.331 332.47 305.226 301.49 385.219 316.81 370.816
        327.724 345.641 330.861 304.222 300.533 380.633 315.143 367.655
        325.674 343.952 329.253 303.217 299.576 376.047 313.477 364.493
        323.624 342.262 327.644 302.212 298.619 371.462 311.81 361.332
        321.574 340.573 326.035 301.208 297.662 366.876 310.144 358.171
        319.524 338.884 324.426 300.203 296.705 362.29 308.477 355.01
        317.475 337.194 322.818 299.198 295.748 357.704 306.811 351.849
        315.425 335.505 321.209 298.194 294.791 353.118 305.144 348.687
        257.906 248.524 258.359 258.816 270.533 257.535 271.568 255.352
        246.25 225.5 228.09 233.912 247.166 243.788 249.95 227.876;;;
        472.066 481.64 482.919 445.543 454.24 493.479 459.551 491.959
        472.225 481.676 482.749 445.584 454.344 493.357 459.411 492.055
        471.96 481.491 482.818 445.475 454.195 493.258 459.48 491.775
        472.013 481.378 482.547 445.448 454.253 492.917 459.271 491.688
        471.644 481.045 482.514 445.272 454.06 492.601 459.269 491.226
        471.592 480.784 482.144 445.177 454.073 492.046 458.992 490.957
        471.122 480.306 482.011 444.935 453.836 491.517 458.921 490.319
        470.967 479.9 481.545 444.775 453.805 490.754 458.577 489.873
        470.399 479.282 481.314 444.469 453.528 490.021 458.44 489.065
        470.143 478.735 480.754 444.245 453.453 489.062 458.034 488.448
        412.83 385.496 408.898 396.216 420.775 398.593 418.838 391.403
        382.756 346.104 361.951 358.648 384.393 368.631 383.055 347.092;;;
        824.403 820.491 834.202 792.156 799.392 829.418 810.472 835.103
        824.687 820.231 834.318 791.907 799.456 829.449 810.806 835.517
        824.344 820.389 834.128 792.103 799.357 829.316 810.43 834.999
        824.568 820.029 834.169 791.802 799.386 829.246 810.72 835.308
        824.165 820.086 833.906 791.945 799.252 829.011 810.301 834.686
        824.329 819.625 833.873 791.592 799.246 828.84 810.548 834.889
        823.868 819.581 833.537 791.682 799.077 828.505 810.088 834.165
        823.972 819.021 833.43 791.277 799.035 828.234 810.291 834.263
        823.454 818.876 833.02 791.314 798.832 827.801 809.79 833.438
        823.498 818.22 832.84 790.859 798.755 827.431 809.948 833.432
        760.999 722.525 750.82 730.266 751.547 741.071 757.277 733.203
        704.144 657.649 680.825 672.359 695.969 682.916 699.526 661.205;;;
        1203.56 1196.9 1206.46 1174.44 1182.11 1200.92 1190.8 1204.63
        1203.98 1197.29 1206.74 1174.7 1181.75 1200.63 1190.63 1205.42
        1203.52 1196.83 1206.4 1174.39 1182.08 1200.87 1190.77 1204.56
        1203.9 1197.16 1206.63 1174.62 1181.69 1200.51 1190.56 1205.29
        1203.4 1196.64 1206.24 1174.27 1181.98 1200.69 1190.66 1204.36
        1203.73 1196.9 1206.42 1174.45 1181.56 1200.28 1190.42 1205.02
        1203.19 1196.31 1205.97 1174.06 1181.82 1200.4 1190.48 1204.03
        1203.48 1196.51 1206.09 1174.2 1181.37 1199.93 1190.2 1204.62
        1202.9 1195.86 1205.59 1173.77 1181.6 1199.99 1190.23 1203.56
        1203.15 1195.99 1205.66 1173.86 1181.11 1199.46 1189.92 1204.08
        1148.01 1120.91 1136.68 1118.24 1134.16 1132.86 1139.26 1124.48
        1088.89 1055.95 1069.59 1060.2 1078.1 1072.69 1080.79 1053.65;;;
        1466.14 1462.23 1464.87 1447.75 1452.8 1462.02 1457.12 1463.51
        1465.84 1461.89 1464.64 1447.63 1452.62 1461.75 1456.77 1463.27
        1466.12 1462.19 1464.84 1447.73 1452.78 1461.99 1457.09 1463.48
        1465.79 1461.83 1464.58 1447.58 1452.57 1461.69 1456.73 1463.19
        1466.04 1462.09 1464.75 1447.65 1452.71 1461.9 1457.02 1463.37
        1465.69 1461.69 1464.45 1447.48 1452.49 1461.57 1456.63 1463.05
        1465.92 1461.92 1464.59 1447.52 1452.61 1461.75 1456.9 1463.19
        1465.54 1461.49 1464.27 1447.32 1452.35 1461.39 1456.49 1462.83
        1465.74 1461.68 1464.37 1447.34 1452.45 1461.53 1456.74 1462.94
        1465.34 1461.22 1464.02 1447.11 1452.18 1461.14 1456.3 1462.55
        1430.08 1418.24 1422.23 1411.48 1420.09 1421.92 1422.31 1416.23
        1389.75 1375.7 1378.51 1372.38 1381.98 1381.92 1382.53 1369.94
    ] atol = 1e-1

    # Can we fit models with the basis function approach?
    Random.seed!(51021)
    x₀ = [0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.025, 1.075, 1.125, 1.175, 1.225, 1.275, 1.325, 1.375, 1.425, 1.475, 1.525, 1.575, 1.625, 1.675, 1.725, 1.775, 1.825, 1.875]
    u₀ = [312.0, 261.0, 233.0, 303.0, 252.0, 252.0, 228.0, 242.0, 238.0, 191.0, 135.0, 140.0, 131.0, 79.3, 46.6, 37.3, 46.6, 46.6, 74.6, 65.3, 28.0, 37.3, 14.0, 4.66, 14.0, 0.0, 9.32, 0.0, 28.0, 97.89999999999999, 172.0, 252.0, 368.0, 350.0, 410.0, 331.0, 350.0]
    T = (t, α, p) -> 1.0
    D = (u, β, p) -> β[1] * p[1]
    D′ = (u, β, p) -> 0.0
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    α = Vector{Float64}([])
    β = [301.0] * t_scale / x_scale^2
    γ = [0.044] * t_scale
    T_params = Vector{Float64}([])
    D_params = [1.0]
    R_params = [K, 1.0]
    x, t, u, datgp = EquationLearning.generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
    x_pde = copy(x)
    t_pde = copy(t)
    u_pde = copy(u)
    σ = log.([1e-6, 7std(u)])
    σₙ = log.([1e-6, 7std(u)])
    gp, μ, L = precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, 250, bootstrap_setup)
    gp_setup = GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts=250, μ, L, nugget, gp)
    D = convert(Vector{Function}, [(u, p) -> 1.0])
    D′ = convert(Vector{Function}, [(u, p) -> 0.0])
    R = convert(Vector{Function}, [(u, p) -> u * (1.0 - u / p[1])])
    R′ = convert(Vector{Function}, [(u, p) -> 1.0 - 2u / p[1]])
    D_params = Vector{Float64}([])
    R_params = [K]
    bootstrap_setup = @set bootstrap_setup.B = 15
    Random.seed!(510212323)
    bgp3 = basis_bootstrap_gp(x, t, u, D, D′, R, R′; gp_setup, bootstrap_setup, pde_setup, D_params, R_params, verbose=false)
    @test bgp3.pde_setup === pde_setup
    pde_setup = @set pde_setup.alg = CVODE_BDF(linear_solver=:Band, jac_upper=1, jac_lower=1)
    pde_data3 = boot_pde_solve(bgp3, x_pde, t_pde, u_pde; ICType="data")
    pde_gp3 = boot_pde_solve(bgp3, x_pde, t_pde, u_pde; ICType="gp")
    @test bgp3.diffusionBases ≈ [0.00605992 0.00634079 0.00523824 0.00674552 0.00575088 0.00757705 0.00625447 0.00617705 0.00657531 0.00679501 0.0062141 0.00596063 0.00683456 0.0064146 0.00699783] atol = 1e-3
    @test bgp3.reactionBases ≈ [1.07008 1.047 1.03705 1.06346 1.04723 1.06998 1.05673 1.05721 1.07045 1.07127 1.04543 1.04791 1.05537 1.04437 1.07207] atol = 1e-3
    @test bgp3.gp === gp
    @test bgp3.Xₛ[1:2, [1, 2, 40, 599, 600, 900]] ≈ [-0.0277778 0.00766284 0.291188 0.964559 1.0 1.0
        0.0 0.0 0.0344828 0.655172 0.655172 1.0] atol = 1e-3
    @test bgp3.Xₛⁿ[1:2, [1, 2, 40, 599, 600, 900]] ≈ [0.025 0.0887931 0.599138 1.81121 1.875 1.875
        0.0 0.0 0.0689655 1.31034 1.31034 2.0] atol = 1e-3
    @test bgp3.Xₛⁿ ≈ bgp3.Xₛ .* [maximum(x) - minimum(x), maximum(t) - minimum(t)] .+ [minimum(x), minimum(t)] atol = 1e-3
    @test bgp3.bootₓ ≈ collect(bgp2.bootₓ) atol = 1e-3
    @test bgp3.bootₜ ≈ collect(bgp2.bootₜ) atol = 1e-3
    @test bgp3.D === D
    @test bgp3.R === R
    @test bgp3.D′ == D′
    @test bgp3.R′ === R′
    @test bgp3.D_params === D_params
    @test bgp3.R_params === R_params
    @test bgp3.μ === μ
    @test bgp3.L === L
    @test bgp3.gp_setup === gp_setup
    @test bgp3.bootstrap_setup === bootstrap_setup
    @test pde_data3[[1, 5, 10, 100], [1, 2, 3, 4, 10], [1, 2, 3, 4, 5]] ≈ [
        301.848 301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848 301.848
        301.848 301.848 301.848 301.848 301.848
        228.767 228.767 228.767 228.767 228.767;;;
        436.446 432.01 433.484 433.708 434.846
        435.854 431.449 432.802 433.183 434.324
        433.589 429.247 430.192 431.03 432.197
        359.531 356.114 355.278 358.181 359.248;;;
        617.556 607.771 607.131 613.015 616.009
        617.233 607.466 606.752 612.729 615.725
        615.9 606.103 605.372 611.661 614.547
        527.084 518.093 517.106 523.034 525.76;;;
        831.237 815.692 812.888 824.636 829.466
        831.001 815.465 812.621 824.42 829.251
        829.738 814.198 811.418 823.837 828.673
        729.315 713.927 712.057 722.557 727.222;;;
        1048.92 1029.51 1025.18 1041.45 1047.53
        1048.74 1029.33 1024.97 1041.28 1047.36
        1047.89 1028.22 1023.8 1040.36 1046.55
        946.468 925.83 923.018 937.662 943.942
    ] atol = 1e-1
    @test pde_gp3[[1, 2, 3, 10, 100, 291, 305], [1, 2, 3, 4, 5, 6, 15], [1, 2, 3, 4, 5]] ≈ [
        259.474 301.274 319.403 282.131 333.993 262.262 255.855
        260.014 299.743 317.378 281.612 330.925 262.497 255.708
        260.555 298.211 315.352 281.093 327.858 262.732 255.56
        264.337 287.492 301.175 277.463 306.387 264.376 254.526
        236.779 239.687 239.236 243.366 246.383 248.94 245.786
        54.6224 44.6437 62.05 49.0552 54.6307 49.7532 55.8508
        38.6481 29.8679 50.305 32.143 37.9126 35.2467 46.1549;;;
        407.036 424.076 438.212 418.457 441.018 409.032 394.093
        407.1 423.962 437.461 418.45 440.809 409.018 393.99
        407.03 423.993 438.055 418.416 440.843 409.044 394.102
        406.969 422.361 434.452 417.657 437.429 409.239 394.173
        361.041 360.891 359.138 368.213 366.836 375.444 370.041
        79.6066 64.5904 89.4394 70.5238 78.1774 68.6346 79.2005
        60.1417 46.8109 74.7066 51.2627 57.7675 52.7466 67.1858;;;
        592.539 602.548 612.68 602.404 616.26 597.628 579.568
        592.882 602.736 612.556 602.448 616.772 597.57 579.56
        592.52 602.495 612.575 602.372 616.161 597.627 579.573
        592.491 601.685 610.509 601.804 614.803 597.549 579.666
        527.737 522.861 519.285 534.665 527.84 542.619 535.229
        121.929 100.129 132.727 108.992 117.926 104.126 119.173
        97.577 77.7744 113.694 85.6514 92.3242 85.449 104.328;;;
        809.39 811.868 817.098 817.236 824.382 816.304 797.879
        809.218 812.356 818.229 816.936 823.585 816.317 797.991
        809.366 811.825 817.021 817.205 824.313 816.294 797.876
        808.739 811.492 816.68 816.334 822.233 816.1 797.932
        729.677 719.579 713.937 735.48 723.502 743.025 734.541
        190.802 159.973 200.067 173.992 181.685 165.938 186.342
        160.929 132.396 176.48 146.204 149.508 144.397 169.548;;;
        1031.03 1027.34 1028.95 1036.38 1037.35 1038.42 1022.41
        1031.14 1027.19 1028.64 1036.27 1036.97 1038.53 1022.37
        1031.0 1027.3 1028.89 1036.35 1037.3 1038.4 1022.4
        1030.64 1026.46 1027.48 1035.7 1035.97 1038.18 1022.16
        946.514 931.99 924.618 950.464 935.035 956.621 948.238
        299.047 256.956 302.07 279.201 280.383 268.761 295.034
        264.807 225.195 274.75 248.371 241.942 245.878 278.33
    ] atol = 1e-1
    idx = EquationLearning.compute_valid_pde_indices(u_pde, 500, nodes, weights, bgp3)
    @test idx == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    dr, rr, d, r, diffusionCIs, reactionCIs = density_values(bgp3; level=0.05, diffusion_scales=x_scale^2 / t_scale, reaction_scales=1 / t_scale)
    @test dr ≈ [252.497 264.199 218.26 281.063 239.62 315.711 260.603 257.377 273.971 283.125 258.921 248.359 284.773 267.275 291.576] atol = 1e-1
    @test rr ≈ [0.0445867 0.0436251 0.0432105 0.0443109 0.0436347 0.0445826 0.0440306 0.0440503 0.044602 0.0446364 0.0435596 0.0436627 0.0439738 0.0435154 0.0446695] atol = 1e-1
    @test d == 1
    @test r == 1
    @test diffusionCIs ≈ [225.736 307.264] atol = 1e-1
    @test reactionCIs ≈ [0.0433172 0.0446579] atol = 1e-1
    Du_vals, Ru_vals, u_vals = curve_values(bgp3; level=0.05, x_scale=x_scale, t_scale=t_scale)
    @test Du_vals[1][[1, 2, 4, 100, 191, 500]] ≈ vec([266.48878959790477
        266.48878959790477
        266.48878959790477
        266.48878959790477
        266.48878959790477
        266.48878959790477]) atol = 1e-1
    @test Du_vals[2][[1, 2, 4, 100, 192, 200, 201, 297, 500]] ≈ vec([225.73603304470163
        225.73603304470163
        225.73603304470163
        225.73603304470163
        225.73603304470163
        225.73603304470163
        225.73603304470163
        225.73603304470163
        225.73603304470163]) atol = 1e-1
    @test Du_vals[3][[1, 2, 100, 50, 199, 201, 499, 500]] ≈ vec([307.2636423705275
        307.2636423705275
        307.2636423705275
        307.2636423705275
        307.2636423705275
        307.2636423705275
        307.2636423705275
        307.2636423705275]) atol = 1e-1
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data3, bgp3)
    @test soln_vals_mean[[1, 2, 3, 50, 101, 171, 199, 500], [1, 2, 3, 4, 5]] ≈ [301.848 433.549 611.532 821.779 1037.42
        301.848 433.526 611.54 821.789 1037.15
        301.848 433.408 611.455 821.722 1037.37
        271.971 396.557 580.1 793.204 1011.86
        229.257 356.364 519.848 717.775 931.644
        128.11 192.571 304.829 462.105 659.537
        56.3149 117.379 208.61 343.373 524.403
        366.906 528.366 729.429 935.51 1128.37] atol = 1e-1
    @test soln_vals_lower[[1, 2, 49, 191, 201, 297, 299, 301, 302], :] ≈ [301.848 431.604 606.701 813.143 1025.82
        301.848 431.589 606.724 813.04 1025.52
        266.853 394.021 574.328 784.039 1000.43
        69.0421 134.103 225.368 360.555 540.754
        53.5546 109.687 195.065 321.821 494.385
        31.1051 54.4342 91.3254 148.214 236.765
        29.9753 51.7026 87.8814 143.836 231.473
        28.8454 49.0386 84.4013 139.459 226.26
        28.2805 47.7601 82.6885 137.285 223.654] atol = 1e-1
    @test soln_vals_upper[[1, 2, 100, 101, 102, 499, 497, 391, 201, 500], :] ≈ [301.848 436.027 617.066 830.711 1048.48
        301.848 436.004 617.078 830.498 1048.32
        228.767 359.445 526.687 728.697 945.706
        229.257 358.346 524.969 726.488 943.365
        229.747 357.216 523.197 724.319 941.0
        362.742 531.184 735.731 944.841 1139.17
        354.413 531.179 735.505 944.502 1138.81
        29.5665 100.913 208.471 359.292 552.282
        53.5546 116.341 210.503 348.879 533.916
        366.906 531.161 735.642 945.753 1140.02] atol = 1e-1
    err_CI = error_comp(bgp3, pde_data3, x_pde, t_pde, u_pde)
    @test err_CI ≈ vec([2.04274318873692
    2.9450531117318204]) atol = 1e-1
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp3, bgp3)
    @test soln_vals_mean[[1, 5, 10, 100, 192, 201, 202, 209, 378, 491, 399, 500], :] ≈ [289.325   418.386   599.088  811.138  1028.99
    285.18    418.12    598.904  810.983  1028.85
    279.999   416.987   598.254  810.437  1028.07
    239.84    361.75    524.729  722.501   935.805
     75.4822  137.557   234.752  376.583   563.879
     64.1703  119.842   209.854  343.762   524.532
     62.9135  118.132   207.367  340.383   520.369
     55.3226  107.675   191.627  318.362   492.571
     12.5968   63.3568  142.986  267.017   440.405
    363.652   532.109   724.322  926.672  1119.31
     81.7425  158.376   272.198  428.301   622.211
    353.021   533.698   727.57   930.732  1122.49] atol = 1e-1
    @test soln_vals_lower[1:30, :] ≈ [257.122  395.124  577.628  792.427  1013.33
    257.098  394.98   577.646  792.578  1013.23
    256.971  395.114  577.608  792.399  1013.29
    256.844  394.961  577.605  792.522  1013.16
    256.718  395.084  577.546  792.317  1013.19
    256.591  394.923  577.523  792.412  1013.02
    256.464  395.036  577.444  792.179  1013.02
    256.337  394.867  577.401  792.246  1012.82
    256.211  394.969  577.301  791.987  1012.78
    256.084  394.794  577.238  792.025  1012.54
    255.957  394.885  577.119  791.739  1012.47
    255.83   394.705  577.035  791.749  1012.2 
    255.704  394.786  576.896  791.437  1012.09
    255.577  394.601  576.793  791.418  1011.79
    255.375  394.672  576.635  791.08   1011.64
    255.049  394.474  576.512  791.032  1011.31
    254.724  394.449  576.336  790.668  1011.13
    254.398  394.116  576.193  790.592  1010.76
    254.292  394.07   575.998  790.202  1010.54
    254.244  393.728  575.811  790.096  1010.14
    254.195  393.662  575.54   789.681  1009.89
    254.147  393.313  575.287  789.545  1009.45
    254.078  393.229  574.992  789.105  1009.17
    253.843  392.875  574.715  788.939  1008.7
    253.609  392.774  574.398  788.474  1008.37
    253.374  392.418  574.098  788.278  1007.87
    253.14   392.302  573.76   787.788  1007.51
    252.905  391.945  573.437  787.562  1006.97
    252.671  391.785  573.077  787.013  1006.57
    252.436  391.36   572.733  786.664  1006.0] atol = 1e-1
    @test soln_vals_upper[[1:5..., 391, 392, 400, 500], :] ≈ [332.132   441.002  619.679  829.624  1044.2
    329.248   440.758  619.943  829.495  1044.0
    326.364   440.83   619.595  829.56   1044.16
    323.48    440.417  619.776  829.367  1043.92
    320.595   440.32   619.341  829.367  1044.04
     67.004   135.321  244.189  400.056   598.057
     70.4529  140.352  250.757  408.003   606.501
    101.707   183.437  305.728  472.423   674.771
    370.118   548.693  745.242  950.373  1140.9] atol = 1e-1
    err_CI = error_comp(bgp3, pde_gp3, x_pde, t_pde, u_pde)
    @test err_CI ≈ vec([3.4774856790511137
    6.150873537957717]) atol = 1e-1
    res = compare_AICs(x_pde, t_pde, u_pde, bgp3, bgp3)
    @test res ≈ [1.0 0.0 0.0
        1.0 0.0 0.0] atol = 1e-1
end

#(:diffusionBases, :reactionBases, :gp, :zvals, :Xₛ, :Xₛⁿ, :bootₓ, :bootₜ, :D, :D′, :R, :R′, :D_params, :R_params, :μ, 
#:L, :gp_setup, :bootstrap_setup, :pde_setup)