#####################################################################
## Load the required package
#####################################################################

using EquationLearning   # Load our actual package 
using CSV                # For loading the density data of Jin et al. (2016).
using DataFrames         # For conveniently representing the data
using CairoMakie         # For creating plots
using LaTeXStrings       # For adding LaTeX labels to plots
using Measures           # Use units to specify some dimensions in plots
using OrderedCollections # For OrderedDict so that dictionaries are sorted by insertion order 
using Random             # For setting seeds 
using GaussianProcesses  # For fitting Gaussian processes 

#####################################################################
## Set some global parameters 
#####################################################################

fontsize = 10
colors = [:black, :blue, :red, :magenta, :green]
alphabet = join('a':'z')
legendentries = OrderedDict("0.0" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[1]),
    "0.5" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[2]),
    "1.0" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[3]),
    "1.5" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[4]),
    "2.0" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[5]))

#####################################################################
## Read in the data from Jin et al. (2016).
#####################################################################

assay_data = Vector{DataFrame}([])
x_scale = 1000.0 # μm ↦ mm 
t_scale = 24.0   # hr ↦ day 
for i = 1:6
    file_name = string("data/CellDensity_", 10 + 2 * (i - 1), ".csv")
    csv_reader = CSV.File(file_name)
    dat = DataFrame(csv_reader)
    dat.Position ./= x_scale
    dat.Dens1 .*= x_scale^2
    dat.Dens2 .*= x_scale^2
    dat.Dens3 .*= x_scale^2
    dat.AvgDens .*= x_scale^2
    dat.Time ./= t_scale
    push!(assay_data, dat)
end
K = 1.7e-3 * x_scale^2 # Cell carrying capacity as estimated from Jin et al. (2016).

#####################################################################
## Figure X: Plotting the density data from Jin et al. (2016).
#####################################################################

assay_plots = Array{Axis}(undef, 3, 2)
jin_assay_data_fig = Figure()

for (k, (i, j)) in enumerate(Tuple.(CartesianIndices(assay_plots)))
    data = assay_data[k]
    assay_plots[i, j] = Axis(jin_assay_data_fig[i, j], xlabel = "Position (mm)", ylabel = "Cell density (cells/mm²)",
        title = "($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign = :left)
    for (s, T) in enumerate(unique(data.Time))
        scatter!(assay_plots[i, j], data.Position[data.Time.==T], data.Dens1[data.Time.==T], color = colors[s], markersize = 3)
        scatter!(assay_plots[i, j], data.Position[data.Time.==T], data.Dens2[data.Time.==T], color = colors[s], markersize = 3)
        scatter!(assay_plots[i, j], data.Position[data.Time.==T], data.Dens3[data.Time.==T], color = colors[s], markersize = 3)
        lines!(assay_plots[i, j], data.Position[data.Time.==T], data.AvgDens[data.Time.==T], color = colors[s])
    end
    hlines!(assay_plots[i, j], K, color = :black)
    ylims!(assay_plots[i, j], 0.0, 2000.0)
    plot_aes!(assay_plots[i, j], fontsize)
end

Legend(jin_assay_data_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (d)", orientation = :horizontal, labelsize = fontsize, titlesize = fontsize, titleposition = :left)
save("figures/jin_assay_data.pdf", jin_assay_data_fig, px_per_unit = 2)

#####################################################################
## Figure X: Plotting the Gaussian processes fit to the data from Jin et al. (2016).
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig = Figure()
gp_plots = Array{Axis}(undef, 3, 2)

for (k, (i, j)) in enumerate(Tuple.(CartesianIndices(gp_plots)))
    data = assay_data[k]
    x = repeat(data.Position, outer = 3)
    t = repeat(data.Time, outer = 3)
    u = vcat(data.Dens1, data.Dens2, data.Dens3)
    gp_dat = EquationLearning.fit_GP(x, t, u)
    μ, Σ = predict_f(gp_dat, [EquationLearning.scale_unit(vec(data.Position)'); EquationLearning.scale_unit(vec(data.Time)')])
    lower = μ .- 2sqrt.(Σ)
    upper = μ .+ 2sqrt.(Σ)
    gp_plots[i, j] = Axis(jin_assay_data_gp_bands_fig[i, j], xlabel = "Position (mm)", ylabel = "Cell density (cells/mm²)",
        title = "($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign = :left)
    for (s, T) in enumerate(unique(t))
        scatter!(gp_plots[i, j], x[t.==T], u[t.==T], color = colors[s], markersize = 3)
        lines!(gp_plots[i, j], data.Position[data.Time.==T], μ[data.Time.==T], color = colors[s])
        band!(gp_plots[i, j], data.Position[data.Time.==T], upper[data.Time.==T], lower[data.Time.==T], color = (colors[s], 0.5))
    end
    hlines!(gp_plots[i, j], K, color = :black)
    ylims!(gp_plots[i, j], 0.0, 2000.0)
    plot_aes!(gp_plots[i, j], fontsize)
end

Legend(jin_assay_data_gp_bands_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (d)", orientation = :horizontal, labelsize = fontsize, titlesize = fontsize, titleposition = :left)
save("figures/jin_assay_data_gp_plots.pdf", jin_assay_data_gp_bands_fig, px_per_unit = 2)

#####################################################################
## Figure X: Plotting the space-time diagram for the Gaussian process.
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig_spacetime = Figure()
spacetime_plots = Array{Axis}(undef, 3, 2)

for (k, (i, j)) in enumerate(Tuple.(CartesianIndices(spacetime_plots)))
    data = assay_data[k]
    x = repeat(data.Position, outer = 3)
    t = repeat(data.Time, outer = 3)
    u = vcat(data.Dens1, data.Dens2, data.Dens3)
    gp_dat = EquationLearning.fit_GP(x, t, u; num_restarts = 50)
    x_rng  = LinRange(extrema(x)..., 200)
    t_rng = LinRange(extrema(t)..., 200)
    X_rng = repeat(x_rng, outer = length(t_rng))
    T_rng = repeat(t_rng, inner = length(x_rng))
    x̃_rng = EquationLearning.scale_unit(X_rng)
    t̃_rng = EquationLearning.scale_unit(T_rng)
    Xₛ = [vec(x̃_rng)'; vec(t̃_rng)']
    μ, _ = predict_f(gp_dat, Xₛ)
    spacetime_plots[i, j] = Axis(jin_assay_data_gp_bands_fig_spacetime[i, j], xlabel = "Position (mm)", ylabel = "Time (d)",
        title = "($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign = :left)
    heatmap!(spacetime_plots[i, j], X_rng, T_rng, μ; colorrange = (0.0, 2000.0))
    plot_aes!(spacetime_plots[i, j], fontsize)
end

cb = Colorbar(jin_assay_data_gp_bands_fig_spacetime[1:3, 3])
cb.colorrange = (0.0, 2000.0)
cb.label = "Cell density (cells/mm²)"
save("figures/jin_assay_data_spacetime_plots.pdf", jin_assay_data_gp_bands_fig_spacetime, px_per_unit = 2)
 