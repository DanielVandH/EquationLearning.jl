"""
    AIC(bgp::Union{BootResults, BasisBootResults}; correct = true)

Computes all the AIC values for the results in `bgp`. A small-sample size correction is used 
if `correct = true`. The formulas used are given in Eq. 6 (uncorrected) or Eq. 17 (corrected) of
Banks and Joyner (2017) [https://doi.org/10.1016/j.aml.2017.05.005].
"""
function AIC(bgp::Union{BootResults,BasisBootResults}, x, t, u; correct = true, pde_solns = nothing)
    ## Number of model parameters and bootstrap iterations
    tt = size(bgp.delayBases, 1)
    d = size(bgp.diffusionBases, 1)
    r = size(bgp.reactionBases, 1)
    κ = tt + d + r
    B = bgp.bootstrap_setup.B
    ## Do we need to re-compute the PDE solutions?
    if isnothing(pde_solns)
        pde_solns = boot_pde_solve(bgp, x, t, u; ICType = "gp")
    end
    ## Now compute all the sum of squared errors for each bootstrap iterate 
    time_values = Array{Bool}(undef, length(t), length(bgp.pde_setup.δt))
    closest_idx = Vector{Vector{Int64}}(undef, length(bgp.pde_setup.δt))
    iterate_idx = Vector{Vector{Int64}}(undef, length(bgp.pde_setup.δt))
    for j = 1:length(unique(t))
        @views time_values[:, j] .= t .== bgp.pde_setup.δt[j]
        @views closest_idx[j] = searchsortednearest.(Ref(bgp.pde_setup.meshPoints), x[time_values[:, j]]) # Use Ref() so that we broadcast only on x[time_values[:, j]] and not the mesh points
        @views iterate_idx[j] = findall(time_values[:, j])
    end
    errs = Vector{Float64}(undef, B)
    N = length(x)
    store_err = Vector{Float64}(undef, N)
    for b in 1:B
        idx = 1
        for j in 1:length(unique(t))
            for (k, i) in enumerate(iterate_idx[j])
                exact = u[i]
                approx = pde_solns[closest_idx[j][k], b, j]
                store_err[idx] = (exact - approx)^2
                idx += 1
            end
        end
        errs[b] = sum(store_err)
    end
    ## Now compute all the AIC values 
    AICs = N * log.(errs ./ N) .+ 2 * (κ .+ 1)
    ## Do we need a correction?
    if correct
        AICs .+= 2 * (κ .+ 1) .* (κ .+ 2) ./ (N - κ)
    end
    return AICs
end

"""
    classify_Δᵢ(Δᵢ::Float64)

Classifies `Δᵢ`, the AICᵢ difference AICᵢ - AICₘᵢₙ, based on the guidelines by Burnham and Anderson, 2004:
- `Δᵢ ≤ 3.0`: Returns `1`, meaning there is substantial evidence that this model is the optimal model.
- `3.0 < Δᵢ ≤ 8.0`: Returns `2`, meaning there is considerably less evidence in favour of this model being optimal.
- `Δᵢ > 8.0`: Returns `3`, meaning there is essentially no evidence that this model is optimal.
"""
function classify_Δᵢ(Δᵢ::Float64)
    if Δᵢ ≤ 3.0
        return 1
    elseif 3.0 < Δᵢ ≤ 8.0
        return 2
    elseif Δᵢ > 8.0
        return 3
    end
end

"""
    compare_AICs(AICs::Float64...)

Compares the AIC values in `AICs...`. See also [`classify_Δᵢ`](@ref) and [`AIC`](@ref).
"""
function compare_AICs(AICs::Float64...)
    min_AIC = minimum(AICs)
    Δᵢ = AICs .- min_AIC
    return classify_Δᵢ.(Δᵢ)
end

"""
    compare_AICs(AICs::Vector{Float64}...)

Compares many AICs by comparing entry-wise. The results are averaged. Assumes that each AIC is of equal length, 
otherwise computes only up to the minimum length.
"""
function compare_AICs(AICs::Vector{Float64}...)
    num_models = length(AICs)
    results = Matrix{Int64}(zeros(num_models, 3))
    B = minimum(length.(AICs))
    num_comparisons = B
    AICs = [AICs[j][i] for i in 1:B, j in 1:num_models]
    for AIC in eachrow(AICs)
        AIC_res = compare_AICs(AIC...)
        for (i, Δᵢ_result) in enumerate(AIC_res)
            @inbounds results[i, Δᵢ_result] += 1
        end
    end
    return results / num_comparisons
end

"""
    compare_AICs(x, t, u, models::Union{BootResults, BasisBootResults}...; correct = true)

Compare several bootstrapped models using AIC.
"""
function compare_AICs(x, t, u, models::Union{BootResults,BasisBootResults}...; correct = true)
    AICs = [AIC(bgp, x, t, u; correct = correct) for bgp in models]
    return compare_AICs(AICs...)
end
