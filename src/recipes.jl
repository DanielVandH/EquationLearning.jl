#####################################################################
## Script description: recipes.jl 
##
## Contains certain utility functions to help with plotting. Despite 
## the name, we do not make use of the @recipes macro from 
## RecipesBase.jl, but in the future we may implement this feature to
## remove the dependency on Plots.jl.
##
## The following functions are defined:
##  - plot_aes!: Change the font size of a plot and add a box.
##  - dens_plot: Create a density plot for some given data.
##  - compute_ribbon_features: Compute the mean and lower/upper limits 
##      for a confidence interval to be used for plotting.
##  - plot_ribbon_with_mean: Plots the results from
##      compute_ribbon_features.
#####################################################################

"""
    plot_aes!(p::AbstractPlot, size)

A convenience function for setting the font size of the plot features consistently.
A box is also placed around the plot.

# Arguments 
- `p`: A plot object to update the font size of; updated in-place.
- `size`: The new font size.

# Outputs
The plot `p` is updated in-place with the new font size and box.
"""
function plot_aes!(p, size)
    plot!(p, titlefontsize = size, tickfontsize = size, legendfontsize = size,
        guidefontsize = size, legendtitlefontsize = size,
        framestyle = :box)
end

"""
    dens_plot(x, CI, xl, sub_lab; size = 13, plot_kwargs...)

Creates density plots for the data in `x` with a confidence region indicated by `CI`.

# Arguments 
- `x`: A vector of values to create the density plot over.
- `CI`: A 2-vector which gives the lower and upper bounds.
- `xlab`: The label to use on the x-axis.
- `sub_lab`: The letter to use for labelling the plot's title.

# Keyword Arguments 
- `size = 13`: The font size to use for the plot. See [`plot_aes!`](@ref).
- `plot_kwargs...`: Other keyword arguments to use for `plot`.

# Outputs 
- `p`: The density plot.
"""
function dens_plot(x, CI, xlab, sub_lab; size = 13, plot_kwargs...)
    @assert length(CI) == 2 "The confidence interval must have two entries only."
    @assert issorted(CI) "The confidence interval must be given in the form (L, U) with L < U."
    p = density(x, label = false, linecolor = :black; plot_kwargs...)
    x_dat = p[1][1][:x]
    y_dat = p[1][1][:y]
    CI_range = CI[1] .< x_dat .< CI[2]
    plot!(p, x_dat[CI_range], repeat([-1e-4], count(CI_range)), fillrange = y_dat[CI_range], # 1e-4 so the thick line doesn't show at the bottom
        fillalpha = 0.35, ylim = (0, 1.1 * maximum(y_dat)), xlabel = xlab, ylabel = "Probability density",
        label = false, color = :blue,
        title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", sub_lab, CI[1], CI[2]),
        titlelocation = :left)
    plot_aes!(p, size)
    return p
end

"""
    compute_ribbon_features(x; level = 0.05)

Computes features for a confidence interval plot for some data `x`.

# Arguments 
- `x`: The data to use for plotting.

# Keyword Arguments 
- `level = 0.05`: The significance level for computing the credible intervals for the parameter values. 

# Outputs 
- `x_mean`: The mean of each row of `x`.
- `x_lower`: The `100(`level/2`)%` quantile for each row of `x`.
- `x_upper`: The `100(`1-level/2`)%` quantile for each row of `x`.
"""
function compute_ribbon_features(x; level = 0.05)
    x_mean = [mean(r) for r in eachrow(x)]
    x_lower = [quantile(r, level / 2) for r in eachrow(x)]
    x_upper = [quantile(r, 1 - level / 2) for r in eachrow(x)]
    return x_mean, x_lower, x_upper
end

"""
    plot_ribbon_with_mean(x, y_mean, y_lower, y_upper; plot_kwargs...)
   
Plots ribbon features from [`compute_ribbon_features`](@ref) for some data `y` corresponding to points `x`.

# Arguments
- `x`: The values corresponding to each point of `y`.
- `y_mean`: The mean of each row of `y` used with [`compute_ribbon_features`](@ref).
- `y_lower`: The `100(`level/2`)%` quantile for each row of `y` used with [`compute_ribbon_features`](@ref).
- `y_upper`: The `100(`1-level/2`)%` quantile for each row of `y` used with [`compute_ribbon_features`](@ref).

# Keyword Arguments 
- `plot_kwargs...`: Other keyword arguments to use for `plot`.

# Outputs 
- `p`: The plotted ribbon.
"""
function plot_ribbon_with_mean(x, y_mean, y_lower, y_upper; plot_kwargs...)
    p = plot(x, y_mean; plot_kwargs...)
    plot!(p, x, y_lower, fillrange = y_upper, fillalpha = 0.35, color = :blue; label = false)
    return p
end