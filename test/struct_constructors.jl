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