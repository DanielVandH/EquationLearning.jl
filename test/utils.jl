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