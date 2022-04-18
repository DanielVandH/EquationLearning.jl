using EquationLearning
using Test
using Random
using StatsBase
using GaussianProcesses
using Distributions
using LinearAlgebra

@testset "Utilities" begin
    ## Is scale_unit working?
    x = [1.0, 4.0, 1.2, 4.0, 6.0, 10.0]
    scaled_x = (x .- minimum(x)) / (maximum(x) - minimum(x))
    @test scaled_x == EquationLearning.scale_unit(x)

    x = [-1.0, -4.0, 0.0, 0.0, 4.0, 2.2, 5.0]
    scaled_x = (x .- minimum(x)) / (maximum(x) - minimum(x))
    @test scaled_x == EquationLearning.scale_unit(x)

    ## Is searchsortednearest working?
    a = [0.2, -0.4, 5.0, 4.0, 1.0, 0.0]
    sort!(a)
    x = 0.01
    @test EquationLearning.searchsortednearest(a, x) == 2

    rev = true
    reverse!(a)
    @test EquationLearning.searchsortednearest(a, x; rev) == 5

    ## Is data_thresholder working?
    τ₁ = 0.01
    τ₂ = 0.05
    t = LinRange(-2π, 2π, 100)
    f = sin.(t)
    fₜ = cos.(t)
    minfτ = minimum(abs.(f)) * τ₁
    maxfτ = maximum(abs.(f)) * (1 - τ₁)
    minfₜτ = minimum(abs.(fₜ)) * τ₂
    maxfₜτ = maximum(abs.(fₜ)) * (1 - τ₂)
    idx₁ = minfτ .≤ abs.(f) .≤ maxfτ
    idx₂ = minfₜτ .≤ abs.(fₜ) .≤ maxfₜτ
    idx₃ = f .≥ 0.0
    idx = findall(idx₁ .& idx₂ .& idx₃)
    @test idx == EquationLearning.data_thresholder(f, fₜ, (τ₁, τ₂))

    ## Is compute_ribbon_features working?
    Random.seed!(29291)
    x = randn(100, 50)
    x_mean = zeros(100)
    x_lower = zeros(100)
    x_upper = zeros(100)
    for i in 1:100
        x_mean[i] = mean(x[i, :])
        x_lower[i] = quantile(x[i, :], 0.025)
        x_upper[i] = quantile(x[i, :], 0.975)
    end
    xm, xl, xu = EquationLearning.compute_ribbon_features(x)
    @test (xm ≈ x_mean) && (x_lower ≈ xl) && (xu ≈ x_upper)

    for i in 1:100
        x_mean[i] = mean(x[i, :])
        x_lower[i] = quantile(x[i, :], 0.25)
        x_upper[i] = quantile(x[i, :], 0.75)
    end
    xm, xl, xu = EquationLearning.compute_ribbon_features(x; level=0.5)
    @test (xm ≈ x_mean) && (x_lower ≈ xl) && (xu ≈ x_upper)
end

@testset "Gaussian Processes" begin
    # Are we computing kernel derivatives correctly?
    x₁ = 0.571
    t₁ = 0.05
    x₂ = 0.58
    t₂ = 0.1
    ℓ₁ = 0.2
    ℓ₂ = 5.0
    test_vals = [-9 // 40,
        -39919 // 1600,
        -1 // 500,
        1 // 500,
        9999 // 250000,
        -9 // 20000,
        -39919 // 800000,
        9 // 40,
        -9 // 20000,
        39919 // 1600,
        -1079271 // 64000,
        -39919 // 1600,
        39919 // 800000,
        1079271 / 64000,
        4780566561 / 2560000]
    functions = [EquationLearning.dkxⱼ,
        EquationLearning.d²kxⱼ,
        EquationLearning.dktⱼ,
        EquationLearning.dktᵢ,
        EquationLearning.d²ktᵢtⱼ,
        EquationLearning.d²ktᵢxⱼ,
        EquationLearning.d³tᵢxⱼ²,
        EquationLearning.dkxᵢ,
        EquationLearning.d²kxᵢtⱼ,
        EquationLearning.d²kxᵢxⱼ,
        EquationLearning.d³kxᵢxⱼ²,
        EquationLearning.d²kxᵢ,
        EquationLearning.d³kxᵢ²tⱼ,
        EquationLearning.d³kxᵢ²xⱼ,
        EquationLearning.d⁴kxᵢ²xⱼ²]
    for (val, func) in tuple.(test_vals, functions)
        @test val ≈ func(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    end

    # Is fit_GP working correctly?
    Random.seed!(29921)
    d = 2
    n = 50
    x = rand(d, n)
    y = vec(sin.(2π * x[1, :]) .* sin.(2π * x[2, :])) + 0.05 * rand(n)
    mZero = MeanZero()
    kern = SE([0.0, 0.0], 0.0)
    gp = GP(x, y, mZero, kern, -2.0)
    optimize!(gp)
    ℓ₁, ℓ₂ = gp.kernel.iℓ2 .^ (-1 / 2)
    σ = gp.kernel.σ2^(1 / 2)
    σₙ = exp(gp.logNoise.value)
    gp_eql = fit_GP(x[1, :], x[2, :], y; σ=log.([1e-7, 7std(y)]), σₙ=log.([1e-7, 7std(y)]))
    ℓ₁eql, ℓ₂eql = gp_eql.kernel.iℓ2 .^ (-1 / 2)
    σeql = gp_eql.kernel.σ2^(1 / 2)
    σₙeql = exp(gp_eql.logNoise.value)
    truevec = [ℓ₁, ℓ₂, σ, σₙ]
    newvec = [ℓ₁eql, ℓ₂eql, σeql, σₙeql]
    discrep = norm(truevec .- newvec)
    @test discrep ≈ 0.017896364999028934

    # Is compute_joint_GP working correctly?
    X = [-1.5 3.0; 0.2 0.6]
    X̃ = [1.2 0.8 2.3; 0.11 0.5 1.4]
    y = [1.0, 0.2]
    ℓ₁ = 2.0
    ℓ₂ = 2.1
    σ = 3.0
    σₙ = 0.1
    gp = GPE(X, y, MeanZero(), SE(log.([ℓ₁, ℓ₂]), log(σ)), log(σₙ))
    L = [2.03561976 1.986637901 0.552986054 -0.2087404649 -0.3565811542 -0.4304071588 -0.3748636561 0.06464575556 -1.076443783 -1.14919821 -1.163860184 -0.1680188118
        0 0.6609671306 0.8307435864 1.241133639 1.207746504 0.7093046849 -0.4653246872 -0.384666246 -0.4871312209 -0.007595118481 -0.3638934901 0.2159403955
        0 0 1.070718784 0.3690295017 0.359942857 0.8414490716 0.2531409219 0.4451578465 -0.09215346392 -0.2709666585 0.01859738077 -0.8505777325
        0 0 0 0.523728437 0.3533370362 -0.5226064357 0.4301290088 0.4166109414 -0.3748379081 -0.2151278959 -0.002869323855 -0.3728569164
        0 0 0 0 0.4333874647 0.06479135897 0.3134131168 0.1056564535 -0.3859580455 -0.02874557382 0.06487315167 0.07816450574
        0 0 0 0 0 0.3701664642 -0.02789035193 0.03599379313 -0.07627453881 -0.1863291217 -0.2063943622 0.1180490051
        0 0 0 0 0 0 0.114722075 -0.09841286932 0.1064483427 0.2983409173 0.3010935321 -0.1611578468
        0 0 0 0 0 0 0 0.05585593722 -0.1401048316 0.04635114285 -0.1330305345 -0.06115393196
        0 0 0 0 0 0 0 0 0.3515120769 -0.1617144428 -0.1228906716 0.434121106
        0 0 0 0 0 0 0 0 0 0.1453126659 -0.008063231579 0.1447680618
        0 0 0 0 0 0 0 0 0 0 0.09444731729 0.01268978171
        0 0 0 0 0 0 0 0 0 0 0 0.249866459] |> Transpose
    μ = [0.4727059514; 
        0.567134272;
        0.2436616807;
        0.01683591794;
        -0.03253925365;
        -0.05661397357;
        -0.2299122164;
        -0.2511710948;
        -0.1113106153;
        0.07713819359;
        0.04385192289;
        0.06585906114]
    M, CL = compute_joint_GP(gp, X̃)
    @test M ≈ μ atol=1e-2
    @test maximum(abs.(CL .- L)) ≈ 0.0 atol=1e-1
end