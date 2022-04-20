using EquationLearning
using Test
using Random
using StatsBase
using GaussianProcesses
using Distributions
using LinearAlgebra
using LatinHypercubeSampling
using PreallocationTools
using DifferentialEquations

import Base: ==
function ==(x::PreallocationTools.DiffCache, y::PreallocationTools.DiffCache)
    x.du == y.du && x.dual_du == y.dual_du
end

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
    μ = [0.4727059514
        0.567134272
        0.2436616807
        0.01683591794
        -0.03253925365
        -0.05661397357
        -0.2299122164
        -0.2511710948
        -0.1113106153
        0.07713819359
        0.04385192289
        0.06585906114]
    M, CL = compute_joint_GP(gp, X̃)
    @test M ≈ μ atol = 1e-2
    @test maximum(abs.(CL .- L)) ≈ 0.0 atol = 1e-1
end

@testset "Preallocation Helpers" begin
    # Is bootstrap_grid working correctly?
    x = [1.0, 2.0, 2.2, 3.3]
    t = [0.0, 0.1, 0.2, 1.0]
    bootₓ = LinRange(1.0, 3.3, 8)
    bootₜ = LinRange(0.0, 1.0, 5)
    x_min = 1.0
    x_max = 3.3
    t_min = 0.0
    t_max = 1.0
    x_rng = 2.3
    t_rng = 1.0
    x̃ = [bootₓ..., bootₓ..., bootₓ..., bootₓ..., bootₓ...]
    t̃ = [repeat([bootₜ[1]], 8)...,
        repeat([bootₜ[2]], 8)...,
        repeat([bootₜ[3]], 8)...,
        repeat([bootₜ[4]], 8)...,
        repeat([bootₜ[5]], 8)...]
    X̃ = (x̃ .- x_min) / x_rng
    T̃ = (t̃ .- t_min) / t_rng
    Xₛ = [Matrix(vec(X̃)'); Matrix(vec(T̃)')]
    nₓnₜ = 8 * 5
    unscaled_t̃ = t̃
    bsgridhelp = EquationLearning.bootstrap_grid(x, t, bootₓ, bootₜ)
    @test bsgridhelp == (x_min, x_max, t_min, t_max, x_rng, t_rng, Xₛ, unscaled_t̃, nₓnₜ)

    # Is preallocate_bootstrap working properly?
    α = [Vector{Float64}([]), [1.0], [1.0, 1.0]]
    β = [Vector{Float64}([]), [1.0], [1.0, 1.0]]
    γ = [Vector{Float64}([]), [1.0], [1.0, 1.0]]
    B = 10
    nₓnₜ = 40
    zvals = [nothing, [0.45168576, -0.34111377, -2.19068893, 0.88297305, -2.83692703, 1.60589977, 1.58033345, 0.44821516, 0.91985635, 0.25184911, 0.01046357, 1.48664823, -0.41557217, -0.01611908, -1.76119862, -0.53973156, 0.98574066, -0.53182668, -2.30424464, 0.44486593, -2.47866563, -1.98129802, 0.05616504, -0.47024353, 0.61831159, -0.64635111, -1.95797178, -0.77730970, 0.27522164, 0.59816854, 0.62166375, -0.60659477, 0.16529989, -0.66605799, -0.52070558, -0.20124944, 0.93014127, 0.72684764, -1.51326352, -0.28392614]]
    for α₀ in α
        for β₀ in β
            for γ₀ in γ
                for z in zvals
                    f = zeros(nₓnₜ)
                    fₜ = zeros(nₓnₜ)
                    fₓ = zeros(nₓnₜ)
                    fₓₓ = zeros(nₓnₜ)
                    ffₜfₓfₓₓ = zeros(4nₓnₜ)
                    f_idx = 1:40
                    fₜ_idx = 41:80
                    fₓ_idx = 81:120
                    fₓₓ_idx = 121:160
                    tt = length(α₀)
                    d = length(β₀)
                    r = length(γ₀)
                    delayBases = zeros(tt, B)
                    diffusionBases = zeros(d, B)
                    reactionBases = zeros(r, B)
                    ℓz = zeros(4nₓnₜ)
                    if isnothing(z)
                        zz = zeros(4nₓnₜ, B)
                    else
                        zz = copy(z)
                    end
                    pbsr = EquationLearning.preallocate_bootstrap(nₓnₜ, α₀, β₀, γ₀, B, z)
                    @test pbsr == (f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
                        f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
                        tt, d, r,
                        delayBases, diffusionBases, reactionBases,
                        ℓz, zz)
                end
            end
        end
    end

    # Is preallocate_eqlearn working correctly?
    num_restarts = [1, 10]
    meshPoints = LinRange(1.0, 10.0, 10)
    δt = [0.5, [0.0, 0.5, 1.0, 1.5, 2.0]]
    finalTime = 2.0
    Xₛ = [1.0 5.0 10.0 1.0 5.0 10.0 1.0 5.0 10.0
        0.0 0.0 0.0 1.0 1.0 1.0 2.0 2.0 2.0]
    tt = [0, 1, 2]
    d = [0, 1, 2]
    r = [0, 1, 2]
    nₓnₜ = size(Xₛ, 2)
    X = [-1.5 3.0; 0.2 0.6]
    X̃ = [1.2 0.8 2.3; 0.11 0.5 1.4]
    y = [1.0, 0.2]
    ℓ₁ = 2.0
    ℓ₂ = 2.1
    σ = 3.0
    σₙ = 0.1
    gp = GPE(Xₛ[:, 1:3], ones(3), MeanZero(), SE(log.([ℓ₁, ℓ₂]), log(σ)), log(σₙ))
    Random.seed!(29291)
    for dt in δt
        for restarts in num_restarts
            for TT in tt
                for DD in d
                    for RR in r
                        if TT + DD + RR > 0
                            lowers = Vector{Float64}([zeros(TT)..., zeros(DD)..., zeros(RR)...])
                            uppers = Vector{Float64}([ones(TT)..., ones(DD)..., ones(RR)...])
                            obj_values = zeros(restarts)
                            if restarts == 1
                                stacked_params = mean(hcat(lowers, uppers); dims=2)
                            else
                                Random.seed!(TT + DD + RR)
                                plan, _ = LHCoptim(restarts, length(lowers), 1000)
                                stacked_params = Matrix(scaleLHC(plan, [(1e-5, 1.0 - 1e-5) for _ in 1:length(lowers)])')
                            end
                            N = 10
                            Δx = diff(meshPoints)
                            V = 1 / 2 * [Δx[1]; Δx[1:(N-2)] + Δx[2:(N-1)]; Δx[N-1]]
                            Du = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10)) # We use dualcache so that we can easily integrate automatic differentiation into our ODE solver. The last argument is the chunk size, see the ForwardDiff docs for details. The /10 is just some heuristic I developed based on warnings given by PreallocationTools.
                            D′u = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
                            Ru = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
                            R′u = DiffEqBase.dualcache(zeros(N), trunc(Int64, N / 10))
                            TuP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
                            DuP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
                            D′uP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
                            RuP = DiffEqBase.dualcache(zeros(nₓnₜ), trunc(Int64, nₓnₜ / 10))
                            RuN = DiffEqBase.dualcache(zeros(5), 6)
                            SSEArray = DiffEqBase.dualcache((zeros(length(meshPoints), dt isa Number ? length(0:dt:finalTime) : length(dt))), trunc(Int64, N / 10))
                            Xₛ₀ = Xₛ[2, :] .== 0.0
                            IC1 = zeros(count(Xₛ₀))
                            initialCondition = zeros(N)
                            MSE = DiffEqBase.dualcache(zeros(size(gp.x, 2)), trunc(Int64, size(gp.x, 2) / 10))
                            Random.seed!(TT + DD + RR)
                            preqlrn = EquationLearning.preallocate_eqlearn(restarts, meshPoints, dt, finalTime, Xₛ, TT, DD, RR, nₓnₜ, gp, lowers, uppers)
                            @test (obj_values, stacked_params,
                                N, Δx, V,
                                Du, D′u, Ru, R′u, TuP, DuP, D′uP, RuP, RuN,
                                SSEArray, Xₛ₀, IC1, initialCondition, MSE) == preqlrn
                        end
                    end
                end
            end
        end
    end

    # Is bootstrap_helper working correctly? Since we just tested the intermediate functions above, we can use them here.
    x = [1.0, 2.0, 2.2, 3.3]
    t = [0.0, 0.1, 0.2, 1.0]
    bootₓ = LinRange(1.0, 3.3, 8)
    bootₜ = LinRange(0.0, 1.0, 5)
    α = [Vector{Float64}([]), [1.0], [1.0, 1.0]]
    β = [Vector{Float64}([]), [1.0], [1.0, 1.0]]
    γ = [Vector{Float64}([]), [1.0], [1.0, 1.0]]
    B = [10, 50, 100]
    num_restarts = [1, 10]
    meshPoints = LinRange(1.0, 10.0, 10)
    δt = [0.5, [0.0, 0.5, 1.0, 1.5, 2.0]]
    finalTime = 2.0
    X = [-1.5 3.0; 0.2 0.6]
    X̃ = [1.2 0.8 2.3; 0.11 0.5 1.4]
    y = [1.0, 0.2]
    ℓ₁ = 2.0
    ℓ₂ = 2.1
    σ = 3.0
    σₙ = 0.1
    gp = GPE(Xₛ[:, 1:3], ones(3), MeanZero(), SE(log.([ℓ₁, ℓ₂]), log(σ)), log(σₙ))
    Random.seed!(29291)
    zvals = [nothing, [0.45168576, -0.34111377, -2.19068893, 0.88297305, -2.83692703, 1.60589977, 1.58033345, 0.44821516, 0.91985635, 0.25184911, 0.01046357, 1.48664823, -0.41557217, -0.01611908, -1.76119862, -0.53973156, 0.98574066, -0.53182668, -2.30424464, 0.44486593, -2.47866563, -1.98129802, 0.05616504, -0.47024353, 0.61831159, -0.64635111, -1.95797178, -0.77730970, 0.27522164, 0.59816854, 0.62166375, -0.60659477, 0.16529989, -0.66605799, -0.52070558, -0.20124944, 0.93014127, 0.72684764, -1.51326352, -0.28392614]]
    for α₀ in α
        for β₀ in β
            for γ₀ in γ
                for b in B
                    for restarts in num_restarts
                        for dt in δt
                            for z in zvals
                                x_min, x_max, t_min, t_max, x_rng, t_rng, Xₛ, unscaled_t̃, nₓnₜ = EquationLearning.bootstrap_grid(x, t, bootₓ, bootₜ)
                                f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ, f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx, tt, d, r, delayBases, diffusionBases, reactionBases, ℓz, zvals = EquationLearning.preallocate_bootstrap(nₓnₜ, α₀, β₀, γ₀, b, z)
                                if tt + d + r > 0
                                    lowers = Vector{Float64}([zeros(tt)..., zeros(d)..., zeros(r)...])
                                    uppers = Vector{Float64}([ones(tt)..., ones(d)..., ones(r)...])
                                    Random.seed!(tt+d+r)
                                    obj_values, stacked_params, N, Δx, V, Du, D′u, Ru, R′u, TuP, DuP, D′uP, RuP, RuN, SSEArray, Xₛ₀, IC1, initialCondition, MSE = EquationLearning.preallocate_eqlearn(restarts, meshPoints, dt, finalTime, Xₛ, tt, d, r, nₓnₜ, gp, lowers, uppers)
                                    Random.seed!(tt+d+r)
                                    bshlper = EquationLearning.bootstrap_helper(x, t, bootₓ, bootₜ, α₀, β₀, γ₀, b, restarts, meshPoints, dt, finalTime, gp, lowers, uppers, z)
                                    @test (x_min, x_max, t_min, t_max, x_rng, t_rng, Xₛ, unscaled_t̃, nₓnₜ,
                                        f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
                                        f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
                                        tt, d, r,
                                        delayBases, diffusionBases, reactionBases,
                                        ℓz, zvals,
                                        obj_values, stacked_params,
                                        N, Δx, V,
                                        Du, D′u, Ru, R′u, TuP, DuP, D′uP, RuP, RuN,
                                        SSEArray, Xₛ₀, IC1,
                                        initialCondition, MSE) == bshlper
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # Is basis_bootstrap_helper working correctly?
    x = [1.0, 2.0, 2.2, 3.3]
    t = [0.0, 0.1, 0.2, 1.0]
    bootₓ = LinRange(1.0, 3.3, 8)
    bootₜ = LinRange(0.0, 1.0, 5)
    d = [1, 2, 3]
    r = [1, 2, 3]
    B = [5, 10, 25]
    for dd in d
        for rr in r
            for b in B
                x_min = 1.0
                t_min = 0.0
                x_rng = 2.3
                t_rng = 1.0
                x̃ = [bootₓ..., bootₓ..., bootₓ..., bootₓ..., bootₓ...]
                t̃ = [repeat([bootₜ[1]], 8)...,
                    repeat([bootₜ[2]], 8)...,
                    repeat([bootₜ[3]], 8)...,
                    repeat([bootₜ[4]], 8)...,
                    repeat([bootₜ[5]], 8)...]
                X̃ = (x̃ .- x_min) / x_rng
                T̃ = (t̃ .- t_min) / t_rng
                Xₛ = [Matrix(vec(X̃)'); Matrix(vec(T̃)')]
                nₓnₜ = 8 * 5
                f = zeros(nₓnₜ)
                fₜ = zeros(nₓnₜ)
                fₓ = zeros(nₓnₜ)
                fₓₓ = zeros(nₓnₜ)
                ffₜfₓfₓₓ = zeros(4nₓnₜ)
                f_idx = 1:40
                fₜ_idx = 41:80
                fₓ_idx = 81:120
                fₓₓ_idx = 121:160
                diffusionBases = zeros(dd, b)
                reactionBases = zeros(rr, b)
                ℓz = zeros(4nₓnₜ)
                zvals = zeros(4nₓnₜ, b)
                A = zeros(nₓnₜ, dd + rr)
                @test (x_min, t_min, x_rng, t_rng, Xₛ,
                    f, fₜ, fₓ, fₓₓ, ffₜfₓfₓₓ,
                    f_idx, fₜ_idx, fₓ_idx, fₓₓ_idx,
                    diffusionBases, reactionBases,
                    ℓz, zvals,
                    A) == EquationLearning.basis_bootstrap_helper(x, t, bootₓ, bootₜ, dd, rr, b)
            end
        end
    end
end

@testset "Basis Function Evaluation" begin
    # Is evaluate_basis working correctly?
    coefficients = [1.0, 0.57, -0.3]
    basis = [(u, p) -> u * p[1], (u, p) -> u + u^2 * p[2], (u, p) -> u^p[3]]
    p = [1.3, -3.0, 5.0]
    point = 0.005
    value = 1.0 * basis[1](point, p) + 0.57 * basis[2](point, p) - 0.3 * basis[3](point, p)
    @test EquationLearning.evaluate_basis(coefficients, basis, point, p) == value

    # Is evaluate_basis! working correctly?
    coefficients = [0.3, 6.0, 1.0]
    coefficients = [1.0, 0.57, -0.3]
    basis = [(u, p) -> u * p[1], (u, p) -> u + u^2 * p[2], (u, p) -> u^p[3]]
    p = [1.3, -3.0, 5.0]
    point = [0.05, 0.18, 1.0, 2.0, 5.0]
    val2 = zeros(length(point))
    val2[1] = EquationLearning.evaluate_basis(coefficients, basis, point[1], p)
    val2[2] = EquationLearning.evaluate_basis(coefficients, basis, point[2], p)
    val2[3] = EquationLearning.evaluate_basis(coefficients, basis, point[3], p)
    val2[4] = EquationLearning.evaluate_basis(coefficients, basis, point[4], p)
    val2[5] = EquationLearning.evaluate_basis(coefficients, basis, point[5], p)
    val = zeros(length(point))
    A = zeros(length(point), length(basis))
    EquationLearning.evaluate_basis!(val, coefficients, basis, point, p, A)
    @test val == val2
end