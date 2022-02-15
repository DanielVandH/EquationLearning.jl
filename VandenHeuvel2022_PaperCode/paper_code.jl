#####################################################################
## Load the required package
#####################################################################

using EquationLearning      # Load our actual package 
using CSV                   # For loading the density data of Jin et al. (2016).
using DataFrames            # For conveniently representing the data
using CairoMakie            # For creating plots
using LaTeXStrings          # For adding LaTeX labels to plots
using Measures              # Use units to specify some dimensions in plots
using OrderedCollections    # For OrderedDict so that dictionaries are sorted by insertion order 
using Random                # For setting seeds 
using GaussianProcesses     # For fitting Gaussian processes 
using DifferentialEquations # For differential equations
using StatsBase             # For std
using LinearAlgebra         # For setting number of threads to prevent StackOverflowError
include("set_parameters.jl")

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
LinearAlgebra.BLAS.set_num_threads(1)

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
    EquationLearning.plot_aes!(assay_plots[i, j], fontsize)
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
    EquationLearning.plot_aes!(gp_plots[i, j], fontsize)
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
    x_rng = LinRange(extrema(x)..., 200)
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
    EquationLearning.plot_aes!(spacetime_plots[i, j], fontsize)
end

cb = Colorbar(jin_assay_data_gp_bands_fig_spacetime[1:3, 3])
cb.colorrange = (0.0, 2000.0)
cb.label = "Cell density (cells/mm²)"
save("figures/jin_assay_data_spacetime_plots.pdf", jin_assay_data_gp_bands_fig_spacetime, px_per_unit = 2)

#####################################################################
## Obtain all the outputs for each study
#####################################################################

x_pde_1_1, t_pde_1_1, u_pde_1_1, args_1_1 = set_parameters(1, assay_data[1], 1, x_scale, t_scale)
x_pde_2_1, t_pde_2_1, u_pde_2_1, args_2_1 = set_parameters(2, assay_data[1], 1, x_scale, t_scale)
x_pde_3_1, t_pde_3_1, u_pde_3_1, args_3_1 = set_parameters(3, assay_data[1], 1, x_scale, t_scale)
x_pde_4_1, t_pde_4_1, u_pde_4_1, args_4_1 = set_parameters(4, assay_data[1], 1, x_scale, t_scale)
x_pde_5_1, t_pde_5_1, u_pde_5_1, args_5_1 = set_parameters(5, assay_data[1], 1, x_scale, t_scale)
x_pde_6_1, t_pde_6_1, u_pde_6_1, args_6_1 = set_parameters(6, assay_data[1], 1, x_scale, t_scale)
x_pde_7_1, t_pde_7_1, u_pde_7_1, args_7_1 = set_parameters(7, assay_data[1], 1, x_scale, t_scale)
x_pde_8_1, t_pde_8_1, u_pde_8_1, args_8_1 = set_parameters(8, assay_data[1], 1, x_scale, t_scale)
x_pde_9_1, t_pde_9_1, u_pde_9_1, args_9_1 = set_parameters(9, assay_data[1], 1, x_scale, t_scale)
x_pde_10_1, t_pde_10_1, u_pde_10_1, args_10_1 = set_parameters(10, assay_data[1], 1, x_scale, t_scale)
x_pde_11_1, t_pde_11_1, u_pde_11_1, args_11_1 = set_parameters(11, assay_data[1], 1, x_scale, t_scale)
x_pde_12_1, t_pde_12_1, u_pde_12_1, args_12_1 = set_parameters(12, assay_data[1], 1, x_scale, t_scale)

x_pde_1_2, t_pde_1_2, u_pde_1_2, args_1_2 = set_parameters(1, assay_data[2], 2, x_scale, t_scale)
x_pde_2_2, t_pde_2_2, u_pde_2_2, args_2_2 = set_parameters(2, assay_data[2], 2, x_scale, t_scale)
x_pde_3_2, t_pde_3_2, u_pde_3_2, args_3_2 = set_parameters(3, assay_data[2], 2, x_scale, t_scale)
x_pde_4_2, t_pde_4_2, u_pde_4_2, args_4_2 = set_parameters(4, assay_data[2], 2, x_scale, t_scale)
x_pde_5_2, t_pde_5_2, u_pde_5_2, args_5_2 = set_parameters(5, assay_data[2], 2, x_scale, t_scale)
x_pde_6_2, t_pde_6_2, u_pde_6_2, args_6_2 = set_parameters(6, assay_data[2], 2, x_scale, t_scale)
x_pde_7_2, t_pde_7_2, u_pde_7_2, args_7_2 = set_parameters(7, assay_data[2], 2, x_scale, t_scale)
x_pde_8_2, t_pde_8_2, u_pde_8_2, args_8_2 = set_parameters(8, assay_data[2], 2, x_scale, t_scale)
x_pde_9_2, t_pde_9_2, u_pde_9_2, args_9_2 = set_parameters(9, assay_data[2], 2, x_scale, t_scale)
x_pde_10_2, t_pde_10_2, u_pde_10_2, args_10_2 = set_parameters(10, assay_data[2], 2, x_scale, t_scale)
x_pde_11_2, t_pde_11_2, u_pde_11_2, args_11_2 = set_parameters(11, assay_data[2], 2, x_scale, t_scale)
x_pde_12_2, t_pde_12_2, u_pde_12_2, args_12_2 = set_parameters(12, assay_data[2], 2, x_scale, t_scale)

x_pde_1_3, t_pde_1_3, u_pde_1_3, args_1_3 = set_parameters(1, assay_data[3], 3, x_scale, t_scale)
x_pde_2_3, t_pde_2_3, u_pde_2_3, args_2_3 = set_parameters(2, assay_data[3], 3, x_scale, t_scale)
x_pde_3_3, t_pde_3_3, u_pde_3_3, args_3_3 = set_parameters(3, assay_data[3], 3, x_scale, t_scale)
x_pde_4_3, t_pde_4_3, u_pde_4_3, args_4_3 = set_parameters(4, assay_data[3], 3, x_scale, t_scale)
x_pde_5_3, t_pde_5_3, u_pde_5_3, args_5_3 = set_parameters(5, assay_data[3], 3, x_scale, t_scale)
x_pde_6_3, t_pde_6_3, u_pde_6_3, args_6_3 = set_parameters(6, assay_data[3], 3, x_scale, t_scale)
x_pde_7_3, t_pde_7_3, u_pde_7_3, args_7_3 = set_parameters(7, assay_data[3], 3, x_scale, t_scale)
x_pde_8_3, t_pde_8_3, u_pde_8_3, args_8_3 = set_parameters(8, assay_data[3], 3, x_scale, t_scale)
x_pde_9_3, t_pde_9_3, u_pde_9_3, args_9_3 = set_parameters(9, assay_data[3], 3, x_scale, t_scale)
x_pde_10_3, t_pde_10_3, u_pde_10_3, args_10_3 = set_parameters(10, assay_data[3], 3, x_scale, t_scale)
x_pde_11_3, t_pde_11_3, u_pde_11_3, args_11_3 = set_parameters(11, assay_data[3], 3, x_scale, t_scale)
x_pde_12_3, t_pde_12_3, u_pde_12_3, args_12_3 = set_parameters(12, assay_data[3], 3, x_scale, t_scale)

x_pde_1_4, t_pde_1_4, u_pde_1_4, args_1_4 = set_parameters(1, assay_data[4], 4, x_scale, t_scale)
x_pde_2_4, t_pde_2_4, u_pde_2_4, args_2_4 = set_parameters(2, assay_data[4], 4, x_scale, t_scale)
x_pde_3_4, t_pde_3_4, u_pde_3_4, args_3_4 = set_parameters(3, assay_data[4], 4, x_scale, t_scale)
x_pde_4_4, t_pde_4_4, u_pde_4_4, args_4_4 = set_parameters(4, assay_data[4], 4, x_scale, t_scale)
x_pde_5_4, t_pde_5_4, u_pde_5_4, args_5_4 = set_parameters(5, assay_data[4], 4, x_scale, t_scale)
x_pde_6_4, t_pde_6_4, u_pde_6_4, args_6_4 = set_parameters(6, assay_data[4], 4, x_scale, t_scale)
x_pde_7_4, t_pde_7_4, u_pde_7_4, args_7_4 = set_parameters(7, assay_data[4], 4, x_scale, t_scale)
x_pde_8_4, t_pde_8_4, u_pde_8_4, args_8_4 = set_parameters(8, assay_data[4], 4, x_scale, t_scale)
x_pde_9_4, t_pde_9_4, u_pde_9_4, args_9_4 = set_parameters(9, assay_data[4], 4, x_scale, t_scale)
x_pde_10_4, t_pde_10_4, u_pde_10_4, args_10_4 = set_parameters(10, assay_data[4], 4, x_scale, t_scale)
x_pde_11_4, t_pde_11_4, u_pde_11_4, args_11_4 = set_parameters(11, assay_data[4], 4, x_scale, t_scale)
x_pde_12_4, t_pde_12_4, u_pde_12_4, args_12_4 = set_parameters(12, assay_data[4], 4, x_scale, t_scale)

x_pde_1_5, t_pde_1_5, u_pde_1_5, args_1_5 = set_parameters(1, assay_data[5], 5, x_scale, t_scale)
x_pde_2_5, t_pde_2_5, u_pde_2_5, args_2_5 = set_parameters(2, assay_data[5], 5, x_scale, t_scale)
x_pde_3_5, t_pde_3_5, u_pde_3_5, args_3_5 = set_parameters(3, assay_data[5], 5, x_scale, t_scale)
x_pde_4_5, t_pde_4_5, u_pde_4_5, args_4_5 = set_parameters(4, assay_data[5], 5, x_scale, t_scale)
x_pde_5_5, t_pde_5_5, u_pde_5_5, args_5_5 = set_parameters(5, assay_data[5], 5, x_scale, t_scale)
x_pde_6_5, t_pde_6_5, u_pde_6_5, args_6_5 = set_parameters(6, assay_data[5], 5, x_scale, t_scale)
x_pde_7_5, t_pde_7_5, u_pde_7_5, args_7_5 = set_parameters(7, assay_data[5], 5, x_scale, t_scale)
x_pde_8_5, t_pde_8_5, u_pde_8_5, args_8_5 = set_parameters(8, assay_data[5], 5, x_scale, t_scale)
x_pde_9_5, t_pde_9_5, u_pde_9_5, args_9_5 = set_parameters(9, assay_data[5], 5, x_scale, t_scale)
x_pde_10_5, t_pde_10_5, u_pde_10_5, args_10_5 = set_parameters(10, assay_data[5], 5, x_scale, t_scale)
x_pde_11_5, t_pde_11_5, u_pde_11_5, args_11_5 = set_parameters(11, assay_data[5], 5, x_scale, t_scale)
x_pde_12_5, t_pde_12_5, u_pde_12_5, args_12_5 = set_parameters(12, assay_data[5], 5, x_scale, t_scale)

x_pde_1_6, t_pde_1_6, u_pde_1_6, args_1_6 = set_parameters(1, assay_data[6], 6, x_scale, t_scale)
x_pde_2_6, t_pde_2_6, u_pde_2_6, args_2_6 = set_parameters(2, assay_data[6], 6, x_scale, t_scale)
x_pde_3_6, t_pde_3_6, u_pde_3_6, args_3_6 = set_parameters(3, assay_data[6], 6, x_scale, t_scale)
x_pde_4_6, t_pde_4_6, u_pde_4_6, args_4_6 = set_parameters(4, assay_data[6], 6, x_scale, t_scale)
x_pde_5_6, t_pde_5_6, u_pde_5_6, args_5_6 = set_parameters(5, assay_data[6], 6, x_scale, t_scale)
x_pde_6_6, t_pde_6_6, u_pde_6_6, args_6_6 = set_parameters(6, assay_data[6], 6, x_scale, t_scale)
x_pde_7_6, t_pde_7_6, u_pde_7_6, args_7_6 = set_parameters(7, assay_data[6], 6, x_scale, t_scale)
x_pde_8_6, t_pde_8_6, u_pde_8_6, args_8_6 = set_parameters(8, assay_data[6], 6, x_scale, t_scale)
x_pde_9_6, t_pde_9_6, u_pde_9_6, args_9_6 = set_parameters(9, assay_data[6], 6, x_scale, t_scale)
x_pde_10_6, t_pde_10_6, u_pde_10_6, args_10_6 = set_parameters(10, assay_data[6], 6, x_scale, t_scale)
x_pde_11_6, t_pde_11_6, u_pde_11_6, args_11_6 = set_parameters(11, assay_data[6], 6, x_scale, t_scale)
x_pde_12_6, t_pde_12_6, u_pde_12_6, args_12_6 = set_parameters(12, assay_data[6], 6, x_scale, t_scale)