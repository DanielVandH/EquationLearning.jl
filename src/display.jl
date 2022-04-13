#####################################################################
## Script description: display.jl
##
## This script contains a method for displaying results to 
## the REPL.
##
#####################################################################

import Base: show

function Base.show(io::IO, results::AllResults)
    # Compute
    B = results.bgp.bootstrap_setup.B
    tt = size(results.bgp.delayBases, 1)
    d = size(results.bgp.diffusionBases, 1)
    r = size(results.bgp.reactionBases, 1)
    level = 0.05
    quantiles = [level / 2 1 - level / 2]
    AICCI = quantile(results.AIC, quantiles)
    # Display 
    println(io, "\nBootstrapping results\n")
    @printf(io, "Number of bootstrap samples: %i\n", B)
    @printf(io, "PDE Error: (%.3g, %.3g)\n", results.pde_error[1], results.pde_error[2])
    @printf(io, "AIC: (%.3g, %.3g)\n\n", AICCI[1], AICCI[2])
    if tt > 0
        for i in 1:tt
            @printf(io, "α[%i]: (%.3g, %.3g)\n", i, results.delayCIs[i, 1], results.delayCIs[i, 2])
        end
    end
    for i in 1:d
        @printf(io, "β[%i]: (%.3g, %.3g)\n", i, results.diffusionCIs[i, 1], results.diffusionCIs[i, 2])
    end
    for i in 1:r
        @printf(io, "γ[%i]: (%.3g, %.3g)\n", i, results.reactionCIs[i, 1], results.reactionCIs[i, 2])
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", results::Vector{AllResults})
    # How many parameters are required at most?
    tt = Vector{Int64}(undef, length(results))
    d = Vector{Int64}(undef, length(results))
    r = Vector{Int64}(undef, length(results))
    for i in 1:length(results)
        tt[i] = size(results[i].bgp.delayBases, 1)
        d[i] = size(results[i].bgp.diffusionBases, 1)
        r[i] = size(results[i].bgp.reactionBases, 1)
    end
    ttₘₐₓ = maximum(tt)
    dₘₐₓ = maximum(d)
    rₘₐₓ = maximum(r)
    max_params = ttₘₐₓ + dₘₐₓ + rₘₐₓ
    # Setup the row names 
    row_names = ["Samples", "Error", "AIC"]
    for i in 1:ttₘₐₓ
        push!(row_names, "α[$i]")
    end
    for i in 1:dₘₐₓ
        push!(row_names, "β[$i]")
    end
    for i in 1:rₘₐₓ
        push!(row_names, "γ[$i]")
    end
    row_names = vcat(row_names, "ℙ(E₁)", "ℙ(E₂)", "ℙ(E₃)")
    # Setup the header 
    header = ["", ["Model $i" for i in 1:length(results)]...]
    # Process each model
    level = 0.05
    quantiles = [level / 2 1 - level / 2]
    model_res = Matrix{Any}(undef, 3 + max_params + 3, length(results))
    for i in 1:length(results)
        model_res[1, i] = results[i].bgp.bootstrap_setup.B
        model_res[2, i] = results[i].pde_error
        model_res[3, i] = quantile(results[i].AIC, quantiles)
        for j in 1:ttₘₐₓ
            model_res[j+3, i] = j > tt[i] ? nothing : results[i].delayCIs[j, :]
        end
        for j in 1:dₘₐₓ
            model_res[j+3+ttₘₐₓ, i] = j > d[i] ? nothing : results[i].diffusionCIs[j, :]
        end
        for j in 1:rₘₐₓ
            model_res[j+3+ttₘₐₓ+dₘₐₓ, i] = j > r[i] ? nothing : results[i].reactionCIs[j, :]
        end
    end
    # Model comparisons
    aic_all = [results[i].AIC for i in 1:length(results)]
    aic_comparisons = compare_AICs(aic_all...)
    for i in 1:length(results)
        model_res[(1+3+ttₘₐₓ+dₘₐₓ+rₘₐₓ):end, i] = aic_comparisons[i, :]
    end
    # Setup table 
    table_data = hcat(row_names, model_res)
    pretty_table(table_data, header; formatters=(v, i, j) -> (v isa Union{Float64,Vector{Float64}}) ? round.(v, digits=3) : v)
end