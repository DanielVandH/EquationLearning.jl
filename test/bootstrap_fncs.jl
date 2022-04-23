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
    Random.seed!(23991)
    pde_gp1 = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="gp")
    pde_data1 = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="data")
    @test pde_gp1[1:4, 1:2, 1] ≈ [296.011 301.977
        294.123 299.737
        292.234 297.496
        290.345 295.256] atol = 1e-3
    @test pde_gp1[97:103, 10:13, 2] ≈ [358.011 366.746 367.533 364.326
        357.276 365.851 366.401 363.274
        356.518 364.92 365.249 362.187
        355.734 363.957 364.064 361.068
        354.923 362.956 362.858 359.912
        354.085 361.922 361.62 358.723
        353.218 360.848 360.359 357.496] atol = 1e-2
    @test pde_gp1[393:399, 17, 3:4] ≈ [245.512 399.523
        252.277 407.514
        259.169 415.729
        266.083 423.809
        273.117 432.116
        280.161 440.264
        287.319 448.643] atol = 1e-2
    @test pde_gp1[192:195, 17:20, [5, 1]] ≈ [554.422 553.568 540.997 570.77
        549.63 548.947 536.561 566.356
        544.865 544.363 532.167 561.977
        540.134 539.825 527.821 557.635;;;
        71.4854 72.2517 67.3345 74.6537
        70.0112 70.9689 66.0492 73.2691
        68.537 69.6861 64.7638 71.8846
        67.0628 68.4033 63.4785 70.5] atol = 1e-2
    @test pde_data1[393:399, 10:13, 1] ≈ [35.4939 35.4939 35.4939 35.4939
        39.0778 39.0778 39.0778 39.0778
        43.8411 43.8411 43.8411 43.8411
        48.6043 48.6043 48.6043 48.6043
        53.3675 53.3675 53.3675 53.3675
        58.1307 58.1307 58.1307 58.1307
        62.8939 62.8939 62.8939 62.8939] atol = 1e-3
    @test pde_data1[97:103, 10:13, 2] ≈ [357.688 359.838 357.35 357.952
        356.801 358.811 356.384 356.935
        355.789 357.848 355.306 355.974
        354.88 356.79 354.301 354.927
        353.844 355.792 353.182 353.931
        352.906 354.696 352.133 352.847
        351.839 353.657 350.969 351.811] atol = 1e-2
    @test pde_data1[393:399, 17, 3:4] ≈ [215.386 363.522
        222.168 371.775
        229.257 380.488
        236.251 388.869
        243.556 397.725
        250.747 406.213
        258.249 415.192] atol = 1e-2
    @test pde_data1[2:10, 17:20, [5, 1]] ≈ [1026.48 1028.32 1026.93 1035.34
        1026.4 1028.01 1026.78 1035.21
        1026.4 1028.23 1026.85 1035.26
        1026.28 1027.88 1026.66 1035.08
        1026.23 1028.06 1026.68 1035.09
        1026.06 1027.67 1026.46 1034.87
        1025.97 1027.8 1026.44 1034.83
        1025.77 1027.37 1026.17 1034.57
        1025.64 1027.46 1026.11 1034.49;;;
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
    Random.seed!(23991)
    AICs1 = AIC(bgp1, x_pde, t_pde, u_pde; correct=false)
    Random.seed!(23991)
    AICs2 = AIC(bgp1, x_pde, t_pde, u_pde; correct=true)
    Random.seed!(23991)
    AICs3 = AIC(bgp1, x_pde, t_pde, u_pde; correct=false, pde_solns=pde_data1)
    Random.seed!(23991)
    AICs4 = AIC(bgp1, x_pde, t_pde, u_pde; correct=false, pde_solns=pde_gp1)
    Random.seed!(23991)
    AICs5 = AIC(bgp1, x_pde, t_pde, u_pde; correct=true, pde_solns=pde_data1)
    Random.seed!(23991)
    AICs6 = AIC(bgp1, x_pde, t_pde, u_pde; correct=true, pde_solns=pde_gp1)
    @test AICs1 == AICs4
    @test AICs2 == AICs6
    @test AICs2 ≈ AICs1 .+ 2 * 12 / (200 - 2)
    @test AICs3 ≈ vec([1148.092204452611
        1067.5249800589413
        1032.0366860231716
        1070.976793921534
        1170.2188769940603
        1043.905766631577
        1084.480809147456
        1062.9999925697714
        1041.53040020126
        1206.4768500848252
        1084.2213050270266
        1081.476714176071
        1121.1795747599606
        1133.859620820133
        1050.65488715628
        1118.1077257017516
        1081.727391124466
        1095.404979151756
        1051.419028965671
        1071.6896523941698])
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
        0.618348 0.539515 0.537072 0.62087 0.639958 0.63816 0.671787 0.585295 0.631955 0.49497 0.597889 0.489293 0.622956 0.519537 0.682896] atol = 1e-3
    @test bgp2.diffusionBases ≈ [1.04983 0.91423 1.02901 0.976896 0.874503 0.90945 0.894913 0.879442 0.906306 0.894008 0.93522 0.981373 0.968472 0.94706 0.928471] atol = 1e-3
    @test bgp2.reactionBases ≈ [1.00849 1.01644 1.00108 0.999999 0.996091 0.999452 0.996499 1.00401 0.984982 0.995392 1.00142 1.00949 0.995022 0.999892 1.00255] atol = 1e-3
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
        0.266508 0.232531 0.231478 0.267595 0.275822 0.275047 0.28954 0.252262 0.272372 0.213332 0.25769 0.210885 0.268494 0.223921 0.294328] atol = 1e-3
    @test dr ≈ [599.456 522.026 587.567 557.808 499.341 519.296 510.995 502.161 517.501 510.478 534.011 560.364 552.998 540.771 530.157] atol = 1e-2
    @test rr ≈ [0.0816874 0.0823317 0.0810878 0.0809999 0.0806834 0.0809556 0.0807164 0.0813246 0.0797836 0.0806268 0.0811147 0.0817689 0.0805968 0.0809912 0.0812066] atol = 1e-3
    @test tt == 2
    @test d == 1
    @test r == 1
    @test delayCIs ≈ [-1.51324 -0.825584
        0.211741 0.292652] atol = 1e-2
    @test diffusionCIs ≈ [500.328 595.295] atol = 1e-2
    @test reactionCIs ≈ [0.0800682 0.0821347] atol = 1e-3
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
        0.2747448797109892]) atol = 1e-3
    @test Tu_vals[2][1:10] ≈ vec([0.18055511307460306
        0.18475545986088773
        0.18903098816595704
        0.19338199724375932
        0.19780873622842626
        0.20231140203725612
        0.20689013729255187
        0.21154502826917498
        0.216276102874953
        0.2210833286713381]) atol = 1e-3
    @test Tu_vals[3][1:10] ≈ vec([0.30459222959515697
        0.3090980091366651
        0.3136403633540367
        0.3182187060392382
        0.3228324253520182
        0.32748088393591795
        0.33216341906968677
        0.33687934285458876
        0.34162794243800926
        0.34640848027369]) atol = 1e-3
    @test Du_vals[1][1:10] ≈ vec([536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551
        536.3286413686551]) atol = 1e-3
    @test Du_vals[2][1:10] ≈ vec([500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491
        500.3281824275491]) atol = 1e-3
    @test Du_vals[3][1:10] ≈ vec([595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887
        595.2946378688887]) atol = 1e-3
    @test Ru_vals[1][1:10] ≈ vec([0.24335096986060162
        0.4793280870828808
        0.7144884269946098
        0.948831989595788
        1.1823587748864157
        1.4150687828664923
        1.6469620135360188
        1.8780384668949957
        2.1082981429434207
        2.337741041681296]) atol = 1e-3
    @test Ru_vals[2][1:10] ≈ vec([0.2403783161421765
        0.47347285494130054
        0.7057605937729302
        0.9372415326370656
        1.1679156715337062
        1.3977830104628526
        1.6268435494245046
        1.8550972884186623
        2.082544227445325
        2.309184366504494]) atol = 1e-3
    @test Ru_vals[3][1:10] ≈ vec([0.24658244051776848
        0.4856931106102229
        0.7239761573621563
        0.9614315807735685
        1.1980593808444593
        1.4338595575748292
        1.668832110964678
        1.9029770410140059
        2.1362943477228122
        2.368784031091098]) atol = 1e-3
    @test u_vals[492:500] ≈ vec([1439.9648553014622
        1442.8914487124116
        1445.818042123361
        1448.7446355343106
        1451.67122894526
        1454.5978223562095
        1457.5244157671589
        1460.4510091781083
        1463.3776025890577]) atol = 1e-3
    @test t_vals[39:43] ≈ vec([0.1523046092184369
        0.156312625250501
        0.16032064128256512
        0.16432865731462926
        0.1683366733466934]) atol = 1e-3
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
        463.6130610175516]) atol = 1e-3
    @test Du_vals[2][1:10] ≈ vec([428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782
        428.2807441184782]) atol = 1e-3
    @test Du_vals[3][1:10] ≈ vec([520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358
        520.096716090358]) atol = 1e-3
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
        9.03379894251022]) atol = 1e-3
    @test Ru_vals[2][79:88] ≈ vec([13.4388611407641
        13.581796447996014
        13.724054055447443
        13.865633963118391
        14.006536171008852
        14.146760679118835
        14.286307487448333
        14.425176595997346
        14.563368004765879
        14.700881713753926]) atol = 1e-3
    @test Ru_vals[3][191:200] ≈ vec([27.17482229897456
        27.24700797666704
        27.318463860561167
        27.389189950656935
        27.459186246954346
        27.52845274945341
        27.596989458154116
        27.664796373056465
        27.731873494160457
        27.798220821466092]) atol = 1e-3
    Random.seed!(29991)
    res = compare_AICs(x_pde, t_pde, u_pde, bgp1, bgp2)
    @test res ≈ [0.0 0.0 1.0; 1.0 0.0 0.0]
end