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
##  - pde_results: Plots the PDE solutions for each bootstrap iteration.
##
#####################################################################

"""
density_results(bgp::BootResults; <keyword arguments>)

    Plots the densities for the bootstrapping results in `bgp`.

# Arguments 
- `bgp`: A [`BootResults`](@ref) object which contains the results for the bootstrapping. 

# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 23`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `delay_scales = nothing`: Values that multiply the individual delay parameters.
- `diffusion_scales = nothing`: Values that multiply the individual diffusion parameters. 
- `reaction_scales = nothing`: Values that multiply the individual reaction parameters.
- `diffusion_resolution = (800, 800)`: Resolution for the diffusion figure.
- `reaction_resolution = (800, 800)`: Resolution for the reaction figure.
- `delay_resolution = (800, 800)`: Resolution for the delay fgure.

# Outputs 
- `delayDensityFigure`: A figure of plots containing a density plot for each delay parameter.
- `diffusionDensityFigure`: A figure of plots containing a density plot for each diffusion parameter.
- `reactionDensityFigure`: A figure of plots containing a density plot for each reaction parameter.
"""
function density_results(bgp::BootResults; level = 0.05, fontsize = 23, delay_scales = nothing, diffusion_scales = nothing, reaction_scales = nothing,
        diffusion_resolution = (800, 800), reaction_resolution = (800, 800), delay_resolution = (800, 800))
    # Start with delay
    trv = copy(bgp.delayBases) 
    if !isnothing(delay_scales)
        try
            trv .*= vec(delay_scales)
        catch
            trv = trv*delay_scales 
        end
    end
    tt = size(trv, 1)
    delayCIs = zeros(tt, 2)
    quantiles = [level / 2 1 - level / 2]
    @inbounds @views for j = 1:tt
        delayCIs[j, :] .= quantile(trv[j, :], quantiles)[1:2] # Do [1:2] to transpose it so that we can broadcast
    end

    # Now work on diffusion 
    dr = copy(bgp.diffusionBases)
    if !isnothing(diffusion_scales)
        try
            dr .*= vec(diffusion_scales)
        catch 
            dr = dr*diffusion_scales
        end
    end
    d = size(dr, 1)
    diffusionCIs = zeros(d, 2)
    @inbounds @views for j = 1:d
        diffusionCIs[j, :] .= quantile(dr[j, :], quantiles)[1:2]
    end

    # Now do reaction 
    rr = copy(bgp.reactionBases)
    if !isnothing(reaction_scales)
        try
            rr .*= vec(reaction_scales)
        catch 
            rr = rr*reaction_scales 
        end
    end
    r = size(rr, 1)
    reactionCIs = zeros(r, 2)
    @inbounds @views for j = 1:r
        reactionCIs[j, :] .= quantile(rr[j, :], quantiles)[1:2]
    end

    # Pre-allocate the plots 
    delayDensityAxes = Vector{Axis}(undef, tt)
    diffusionDensityAxes = Vector{Axis}(undef, d)
    reactionDensityAxes = Vector{Axis}(undef, r)
    alphabet = join('a':'z') # For labelling the figures

    # Plot the delay coefficient densities 
    delayDensityFigure = Figure(fontsize = fontsize, resolution = delay_resolution)
    for i = 1:tt
        delayDensityAxes[i] = Axis(delayDensityFigure[1, i], xlabel = L"\alpha_%$i", ylabel = "Probability density",
            title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
            titlealign = :left)
        densdat = kde(trv[i, :])
        lines!(delayDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
        CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
        band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
    end

    # Plot the diffusion coefficient densities 
    diffusionDensityFigure = Figure(fontsize = fontsize, resolution = diffusion_resolution)
    for i = 1:d
        diffusionDensityAxes[i] = Axis(diffusionDensityFigure[1, i], xlabel = L"\beta_%$i", ylabel = "Probability density",
            title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], diffusionCIs[i, 1], diffusionCIs[i, 2]),
            titlealign = :left)
        densdat = KernelDensity.kde(dr[i, :])
        lines!(diffusionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
        CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
        band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
    end

    # Plot the reaction coefficient densities
    reactionDensityFigure = Figure(fontsize = fontsize, resolution = reaction_resolution)
    for i = 1:r
        reactionDensityAxes[i] = Axis(reactionDensityFigure[1, i], xlabel = L"\gamma_%$i", ylabel = "Probability density",
            title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], reactionCIs[i, 1], reactionCIs[i, 2]),
            titlealign = :left)
        densdat = kde(rr[i, :])
        lines!(reactionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
        CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
        band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
    end

    # Return
    return delayDensityFigure, diffusionDensityFigure, reactionDensityFigure
end

"""
    curve_results(bgp::BootResults; <keyword arguments>)

Plots the learned functional forms along with confidence intervals for the bootstrapping results in `bgp`.

# Arguments 
- `bgp`: A [`BootResults`](@ref) object which contains the results for the bootstrapping. 
    
# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 23`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `x_scale = 1.0`: Value used for scaling the spatial data (and all other length units, e.g. for diffusion).
- `t_scale = 1.0`: Value used for scaling the temporal data (and all other time units, e.g. for reaction).

# Outputs
There are no returned values from this function, but there plots created inside R using `RCall`:
- `delayCurvePlots`: A plot containing the learned functional form for the delay function, along with an uncertainty ribbon.
- `diffusionCurvePlots`: A plot containing the learned functional form for the diffusion function, along with an uncertainty ribbon.
- `reactionCurvePlots`: A plot containing the learned functional form for the reaction function, along with an uncertainty ribbon.
"""
function curve_results(bgp::BootResults; level = 0.05, fontsize = 23, x_scale = 1.0, t_scale = 1.0)
    # Setup parameters and grid for evaluation
    trv = bgp.delayBases
    dr = bgp.diffusionBases
    rr = bgp.reactionBases
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
        Du[:, j] .= bgp.D.(u_vals, Ref(dr[:, j]), Ref(bgp.D_params)) * x_scale^2/t_scale
        Ru[:, j] .= bgp.R.(u_vals, Ref(rr[:, j]), Ref(bgp.R_params)) / (x_scale * t_scale)
    end

    # Find lower/upper values for confidence intervals, along with mean curves
    Tu_vals = compute_ribbon_features(Tu; level = level)
    Du_vals = compute_ribbon_features(Du; level = level)
    Ru_vals = compute_ribbon_features(Ru; level = level)

    # Plot delay curves 
    delayCurvePlots = Figure(fontsize = fontsize)
    ax = Axis(delayCurvePlots[1, 1], xlabel = L"t", ylabel = L"T(t)", linewidth = 1.3, linecolor = :blue)
    lines!(ax, t_vals*t_scale, Tu_vals[1])
    band!(ax, t_vals*t_scale, Tu_vals[3], Tu_vals[2], color = (:blue, 0.35))

    # Plot the diffusion curves 
    diffusionCurvePlots = Figure(fontsize = fontsize)
    ax = Axis(diffusionCurvePlots[1, 1], xlabel = L"u", ylabel = L"D(u)", linewidth = 1.3, linecolor = :blue)
    lines!(ax, u_vals/x_scale^2, Du_vals[1])
    band!(ax, u_vals/x_scale^2, Du_vals[3], Du_vals[2], color = (:blue, 0.35))

    # Plot the reaction curves 
    reactionCurvePlots = Figure(fontsize = fontsize)
    ax = Axis(reactionCurvePlots[1, 1], xlabel = L"u", ylabel = L"R(u)", linewidth = 1.3, linecolor = :blue)
    lines!(ax, u_vals/x_scale^2, Ru_vals[1])
    band!(ax, u_vals/x_scale^2, Ru_vals[3], Ru_vals[2], color = (:blue, 0.35))

    # Return 
    return delayCurvePlots, diffusionCurvePlots, reactionCurvePlots
end

"""
    pde_results(x_pde, t_pde u_pde, solns_all, bgp::BootResults; <keyword arguments>)

Plots the solutions to the PDEs corresponding to the bootstrap samples of [`bootstrap_gp`](@ref) using 
the computed solutins in [`boot_pde_solve`](@ref).

# Arguments
- `x_pde`: The spatial data to plot as points.
- `t_pde`: The temporal data to plot as points.
- `u_pde`: The density data to plot as points. 
- `solns_all`: The solutions to the PDEs for each bootstrap iteration. 
- `bgp::BootResults`: A [`BootResults`](@ref) struct containing the results from [`bootstrap_gp`](@ref).

# Keyword Arguments 
- `colors = [:black, :blue, :red, :magenta, :green]`: A list of colors for colouring the solutions at each time.
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 
- `fontsize = 23`: Font size for the plots (to be used in [`plot_aes!`](@ref)).
- `x_scale = 1.0`: Value used for scaling the spatial data (and all other length units, e.g. for diffusion).
- `t_scale = 1.0`: Value used for scaling the temporal data (and all other time units, e.g. for reaction).`

# Outputs 
- `pdeSolutionPlots_BGP`: The plot of the PDE solutions.
"""
function pde_results(x_pde, t_pde, u_pde, solns_all, bgp::BootResults;
    colors = [:black, :blue, :red, :magenta, :green], level = 0.05, fontsize = 23,
    x_scale = 1.0, t_scale = 1.0)
    ## Setup
    N = length(bgp.pde_setup.meshPoints)
    M = length(bgp.pde_setup.δt)
    @assert length(colors) == M "There must be as many provided colors as there are unique time values."
    soln_vals_mean = zeros(N, M)
    soln_vals_lower = zeros(N, M)
    soln_vals_upper = zeros(N, M)
    for j = 1:M
        soln_vals_mean[:, j], soln_vals_lower[:, j], soln_vals_upper[:, j] = compute_ribbon_features(solns_all[:, :, j]; level = level)
    end

    # Initiate axis 
    pdeSolutionPlots_BGP = Figure(fontsize = fontsize)
    ax = Axis(pdeSolutionPlots_BGP[1, 1], xlabel = L"x", ylabel = L"u(x, t)")

    # Plot the lines, ribbons, and data 
    @views for j in 1:M
        lines!(ax, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
        band!(ax, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
        CairoMakie.scatter!(ax, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 7)
    end

    return pdeSolutionPlots_BGP
end