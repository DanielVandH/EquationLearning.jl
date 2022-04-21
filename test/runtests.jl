using EquationLearning
using Test
using Random
using StatsBase
using GaussianProcesses
using Distributions
using LinearAlgebra
using LatinHypercubeSampling
using OrderedCollections
using PreallocationTools
using Setfield
using DifferentialEquations
using Printf
using KernelDensity
using FastGaussQuadrature
using StructEquality
using Optim
using CairoMakie
using Coverage # using Pkg; Pkg.test("EquationLearning", coverage=true). See https://github.com/JuliaCI/Coverage.jl
LinearAlgebra.BLAS.set_num_threads(1)
import Base: ==
function ==(x::PreallocationTools.DiffCache, y::PreallocationTools.DiffCache)
    x.du == y.du && x.dual_du == y.dual_du
end

function ==(x::Missing, y::Missing)
    true
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
                                    Random.seed!(tt + d + r)
                                    obj_values, stacked_params, N, Δx, V, Du, D′u, Ru, R′u, TuP, DuP, D′uP, RuP, RuN, SSEArray, Xₛ₀, IC1, initialCondition, MSE = EquationLearning.preallocate_eqlearn(restarts, meshPoints, dt, finalTime, Xₛ, tt, d, r, nₓnₜ, gp, lowers, uppers)
                                    Random.seed!(tt + d + r)
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

@testset "Struct Constructors" begin
    # Is the GP_Setup constructor working?
    u = [1.0, 5.0, 0.1, 0.2, 0.5]
    ℓₓ = log.([1e-6, 1.0])
    ℓₜ = log.([1e-6, 1.0])
    σ = log.([1e-6, 7std(u)])
    σₙ = log.([1e-6, 7std(u)])
    GP_Restarts = 50
    μ = [missing, [20.0, 30.0, 40.0]]
    L = [missing, LowerTriangular{Float64}([20.0 0.0; 10.0 5.0])]
    nugget = 1e-4
    Xₛ = [1.0 5.0 10.0 1.0 5.0 10.0 1.0 5.0 10.0
        0.0 0.0 0.0 1.0 1.0 1.0 2.0 2.0 2.0]
    gp = [missing, GPE(Xₛ[:, 1:3], ones(3), MeanZero(), SE([0.0, 0.0], 0.0), -2.0)]
    for m in μ
        for ℓ in L
            for gpp in gp
                gpsetup = GP_Setup(ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, m, ℓ, nugget, gpp)
                @test GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ=m, L=ℓ, nugget, gp=gpp) == gpsetup
            end
        end
    end

    # Is the Bootstrap_Setup constructor working?
    bootx = [LinRange(0.0, 10.0, 50), LinRange(-10.0, 10.0, 250)]
    boott = [LinRange(0.0, 5.0, 123), LinRange(0.0, 10.0, 50)]
    b = [2, 5, 10, 1000]
    t = [(0.0, 0.0), (0.3, 0.5)]
    optrestarts = [1, 5]
    constrain = [false, true]
    glsscale = [x -> x, x -> 1, log, exp]
    pdescale = [x -> x, x -> 1, log, exp]
    initweight = [0.0, 10.0]
    showlosses = [false, true]
    for bootₓ in bootx
        for bootₜ in boott
            for B in b
                for τ in t
                    for Optim_Restarts in optrestarts
                        for constrained in constrain
                            for obj_scale_GLS in glsscale
                                for obj_scale_PDE in pdescale
                                    for init_weight in initweight
                                        for show_losses in showlosses
                                            @test Bootstrap_Setup(rand(10), rand(10); bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, init_weight, show_losses) == Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, init_weight, show_losses)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # Is the PDE_Setup constructor working properly?
    for _ in 1:20
        meshPoints = LinRange(extrema(10rand(2))..., 500)
        LHS = rand(3)
        RHS = rand(3)
        finalTime = abs.(100rand())
        r = rand()
        x = rand(10)
        t = rand(5)
        if r < 0.5
            δt = finalTime / 4
        else
            δt = 0:finalTime/4:finalTime
        end
        q = rand()
        if q < 0.5
            alg = nothing
        else
            alg = Tsit5()
        end
        @test PDE_Setup(x, t; meshPoints, LHS, RHS, finalTime, δt, alg) == PDE_Setup(meshPoints, LHS, RHS, finalTime, 0:finalTime/4:finalTime, alg)
    end
end

@testset "Bootstrapping" begin
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
    nₓ = 80
    nₜ = 75
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
    @test all((sum(μ), prod(abs.(log.(abs.(μ))) .^ (0.0001)), median(μ), mean(μ), mean(sin.(μ))) .≈ ((2.358682365154418e6, 91.87561481225711, 325.4003821948756, 98.27843188143409, 0.002763346931410305)))
    @test all((mean(L), median(Symmetric(L)), sum(Symmetric(L)), Symmetric(L)[2, 2], sum(eigen(Symmetric(L)[1:200, 1:200]).values)) .≈ ((0.007124925418994558, 0.0, 10034.10512490004, 3.246503669101823, 28.486899239544083)))

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
    @test bgp1.diffusionBases ≈ Matrix([0.8513680174237094,0.8306630991752704,0.9835242794147602,0.921732832410844,0.9733773297949228,0.9226935020777771,0.7793534011968919,0.8977385045132767,0.8791413689392109,1.0495277282327105,0.9092867495034838,0.8127849598665242,0.980462650553461,0.9303273198133221,0.8430216583525106,0.9850939849138696,0.8851952815813847,1.0023956521556008,0.9860105693350002,0.9218269644490906]')
    @test bgp1.reactionBases ≈ Matrix([1.001282480560487,0.9929224340035784,1.0020561177614897,1.0036890397229268,1.008784895608166,1.0058788209398541,1.006596786696963,1.0063079736385774,1.0045940703842977,1.0044283567887256,1.0011432979208819,0.9909570426487228,1.0072285264928207,1.0107672368252236,0.995563932496905,1.0001536078456275,1.0165344949319828,1.0036696951245252,1.0168349552135956,1.0148477480794655]')
    @test bgp1.gp === gp
    @test mean(bgp1.zvals) ≈ 0.0013533892936884353
    @test bgp1.zvals[2000, 7] ≈ 0.8946445944917903
    @test bgp1.Xₛ[1:2, 1:5] ≈ [-0.0277778 -0.0147679 -0.00175809 0.0112518 0.024261; 0.0 0.0 0.0 0.0 0.0] atol = 1e-6
    @test bgp1.Xₛⁿ[1:2, 4000:4007] ≈ [1.875 0.025 0.0484177 0.0718354 0.0952532 0.118671 0.142089 0.165506; 1.32432 1.35135 1.35135 1.35135 1.35135 1.35135 1.35135 1.35135] atol = 1e-5
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
    @test diffusionCIs ≈ [239.365 309.169] atol=1e-3
    @test reactionCIs ≈ [0.0436432 0.0447345] atol=1e-6

    # Is curve_values working correctly?
    Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp1; level=0.05, x_scale=x_scale, t_scale=t_scale)
    @test Tu_vals[1] == Tu_vals[2] == Tu_vals[3] == repeat([1.0], 500)
    @test unique(Du_vals[1]) ≈ [276.10016409823953]
    @test unique(Du_vals[2]) ≈ [239.36525086105516]
    @test unique(Du_vals[3]) ≈ [309.1691376209211]
    @test Ru_vals[1][[1, 10, 59, 100, 250, 397, 500]] ≈ vec([0.102493  1.00685  5.53705  8.81671  16.8485  18.6745  16.3886]) atol=1e-4
    @test Ru_vals[2][[4, 100, 19, 500, 325]] ≈ [0.40125016693853544, 8.704195093118189, 1.8646702614672126, 16.179425254323252, 18.29139469162819]
    @test Ru_vals[3][[1, 2, 55, 100, 100, 325]] ≈ [0.10371465515802154, 0.20651772630519405, 5.25407588991313, 8.921838300762065, 8.921838300762065, 18.748760108002376]
    @test u_vals[[1, 3, 5, 100, 500]] ≈ [2.3216216643979295, 6.9366378754184215, 11.551654086438912, 230.76492410991227, 1153.7681663140106] 
    @test t_vals[[1, 2, 3, 4, 100, 250, 377]] ≈ [0.0, 0.004008016032064128, 0.008016032064128256, 0.012024048096192385, 0.396793581743487, 0.9979959919839679, 1.5070140280561122]

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

end

#(:delayBases, :diffusionBases, :reactionBases, :gp, 
#:zvals, :Xₛ, :Xₛⁿ, :bootₓ, :bootₜ, :T, :D, :D′, :R, :R′, 
#:D_params, :R_params, :T_params, :μ, :L, :gp_setup, 
#:bootstrap_setup, :pde_setup)