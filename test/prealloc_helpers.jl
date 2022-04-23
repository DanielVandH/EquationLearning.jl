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