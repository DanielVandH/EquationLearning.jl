#####################################################################
## Script description: plot_results.jl.
##
## This script contains functions for plotting results from the 
## equation learning process.
##
## The following functions are defined:
##  - density_results: Plots the densities of the estimates from the 
##      bootstrapping procedure.
##  - curve_results: Plots the learned functional forms for the parameter 
##      estimates from the bootstrapping procedure.
##
#####################################################################


"""
density_results(bgp::BootResults; level = 0.05, fontsize = 13, plot_kwargs...)

    Plots the densities for the bootstrapping results in `bgp`.

# Arguments 
- `bgp`: A [`BootResults`](@ref) object which contains the results for the bootstrapping. 

# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 13`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `plot_kwargs...`: Other keyword arguments to be used in `plot`.

# Outputs 
- `delayDensityPlots`: An array of plots containing a density plot for each delay parameter.
- `diffusionDensityPlots`: An array of plots containing a density plot for each diffusion parameter.
- `reactionDensityPlots`: An array of plots containing a density plot for each reaction parameter.
"""
function density_results(bgp::BootResults; level = 0.05, fontsize = 13, plot_kwargs...)
    # Start with delay
    trv = bgp.delayBases
    tt = size(trv, 1)
    delayCIs = zeros(tt, 2)
    quantiles = [level / 2 1 - level / 2]
    @inbounds @views for j = 1:tt
        delayCIs[j, :] .= quantile(trv[j, :], quantiles)[1:2] # Do [1:2] to transpose it so that we can broadcast
    end

    # Now work on diffusion 
    dr = bgp.diffusionBases
    d = size(dr, 1)
    diffusionCIs = zeros(d, 2)
    @inbounds @views for j = 1:d
        diffusionCIs[j, :] .= quantile(dr[j, :], quantiles)[1:2]
    end

    # Now do reaction 
    rr = bgp.reactionBases
    r = size(rr, 1)
    reactionCIs = zeros(r, 2)
    @inbounds @views for j = 1:r
        reactionCIs[j, :] .= quantile(rr[j, :], quantiles)[1:2]
    end

    # Pre-allocate the plots 
    delayDensityPlots = Array{AbstractPlot}(undef, tt)
    diffusionDensityPlots = Array{AbstractPlot}(undef, d)
    reactionDensityPlots = Array{AbstractPlot}(undef, r)
    alphabet = join('a':'z') # For labelling the figures

    # Plot the delay coefficient densities 
    for i = 1:tt
        delayDensityPlots[i] = dens_plot(trv[i, :], delayCIs[i, :], L"\alpha_%$i", alphabet[i]; size = fontsize, bottom_margin = 2mm, plot_kwargs...) # %$ is string interpolation for LaTeX strings
    end

    # Plot the diffusion coefficient densities 
    for i = 1:d
        diffusionDensityPlots[i] = dens_plot(dr[i, :], diffusionCIs[i, :], L"\beta_%$i", alphabet[i]; size = fontsize, bottom_margin = 2mm, plot_kwargs...)
    end

    # Plot the reaction coefficient densities
    for i = 1:r
        reactionDensityPlots[i] = dens_plot(rr[i, :], reactionCIs[i, :], L"\gamma_%$i", alphabet[i]; size = fontsize, bottom_margin = 2mm, plot_kwargs...)
    end

    # Return
    return delayDensityPlots, diffusionDensityPlots, reactionDensityPlots
end

"""
    curve_results(bgp::BootResults; <keyword arguments>)

Plots the learned functional forms along with confidence intervals for the bootstrapping results in `bgp`.

# Arguments 
- `bgp`: A [`BootResults`](@ref) object which contains the results for the bootstrapping. 
    
# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 13`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `plot_kwargs...`: Other keyword arguments to be used in `plot`.
    
# Outputs
There are no returned values from this function, but there plots created inside R using `RCall`:
- `delayCurvePlots`: A plot containing the learned functional form for the delay function, along with an uncertainty ribbon.
- `diffusionCurvePlots`: A plot containing the learned functional form for the diffusion function, along with an uncertainty ribbon.
- `reactionCurvePlots`: A plot containing the learned functional form for the reaction function, along with an uncertainty ribbon.
"""
function curve_results(bgp::BootResults; level = 0.05, fontsize = 13, plot_kwargs...)
    # Setup parameters and grid for evaluation
    trv = bgp.delayBases
    dr = bgp.diffusionBases
    rr = bgp.reactionBases
    tt = size(trv, 1)
    B = size(dr, 2)
    num_u = 500
    num_t = 500
    u_vals = collect(range(minimum(bgp.gp.y), maximum(bgp.gp.y), length = num_u))
    t_vals = collect(range(minimum(bgp.Xₛⁿ[2, :]), maximum(bgp.Xₛⁿ[2, :]), length = num_t))

    # Evaluate curves 
    Tu = zeros(num_u, B)
    Du = zeros(num_u, B)
    Ru = zeros(num_u, B)
    @views @inbounds for j = 1:B
        Tu[:, j] .= bgp.T.(t_vals, Ref(trv[:, j]), Ref(bgp.T_params)) # Use Ref() so that we only broadcast over the first argument.
        Du[:, j] .= bgp.D.(u_vals, Ref(dr[:, j]), Ref(bgp.D_params))
        Ru[:, j] .= bgp.R.(u_vals, Ref(rr[:, j]), Ref(bgp.R_params))
    end

    # Find lower/upper values for confidence intervals, along with mean curves
    Tu_vals = compute_ribbon_features(Tu; level = level)
    Du_vals = compute_ribbon_features(Du; level = level)
    Ru_vals = compute_ribbon_features(Ru; level = level)

    # Plot delay curves 
    if tt > 0
        delayCurvePlots = plot_ribbon_with_mean(t_vals, Tu_vals...; label = false, xlabel = L"t", ylabel = L"T(t)", linewidth = 1.3, linecolor = :blue, plot_kwargs...)
        plot_aes!(delayCurvePlots, fontsize)
    else
        delayCurvePlots = nothing
    end

    # Plot the diffusion curves 
    diffusionCurvePlots = plot_ribbon_with_mean(u_vals, Du_vals...; xlabel = L"u", ylabel = L"D(u)", linewidth = 1.3, linecolor = :blue, label = false, plot_kwargs...)
    plot_aes!(diffusionCurvePlots, fontsize)

    # Plot the reaction curves 
    reactionCurvePlots = plot_ribbon_with_mean(u_vals, Ru_vals...; xlabel = L"u", ylabel = L"R(u)", label = false, plot_kwargs...)
    plot_aes!(reactionCurvePlots, fontsize)

    # Return 
    return delayCurvePlots, diffusionCurvePlots, reactionCurvePlots
end