#####################################################################
## Load the required package
#####################################################################

using EquationLearning      # Load our actual package 
using DelimitedFiles        # For loading the density data of Jin et al. (2016).
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
using Setfield              # For modifying immutable structs
include("set_parameters.jl")

#####################################################################
## Set some global parameters 
#####################################################################

fontsize = 23
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

function prepare_data(filename) # https://discourse.julialang.org/t/failed-to-precompile-csv-due-to-load-error/70146/2
    data, header = readdlm(filename, ',', header = true)
    df = DataFrame(data, vec(header))
    df_new = identity.(df)
    return df_new
end

assay_data = Vector{DataFrame}([])
x_scale = 1000.0 # μm ↦ mm 
t_scale = 24.0   # hr ↦ day 
for i = 1:6
    file_name = string("data/CellDensity_", 10 + 2 * (i - 1), ".csv")
    dat = prepare_data(file_name)
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
jin_assay_data_fig = Figure(fontsize = fontsize)

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
end

Legend(jin_assay_data_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (d)", orientation = :horizontal, labelsize = fontsize, titlesize = fontsize, titleposition = :left)
save("figures/jin_assay_data.pdf", jin_assay_data_fig, px_per_unit = 2)

#####################################################################
## Figure X: Plotting the Gaussian processes fit to the data from Jin et al. (2016).
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig = Figure(fontsize = fontsize)
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
        band!(gp_plots[i, j], data.Position[data.Time.==T], upper[data.Time.==T], lower[data.Time.==T], color = (colors[s], 0.35))
    end
    hlines!(gp_plots[i, j], K, color = :black)
    ylims!(gp_plots[i, j], 0.0, 2000.0)
end

Legend(jin_assay_data_gp_bands_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (d)", orientation = :horizontal, labelsize = fontsize, titlesize = fontsize, titleposition = :left)
save("figures/jin_assay_data_gp_plots.pdf", jin_assay_data_gp_bands_fig, px_per_unit = 2)

#####################################################################
## Figure X: Plotting the space-time diagram for the Gaussian process.
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig_spacetime = Figure(fontsize = fontsize)
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
end

cb = Colorbar(jin_assay_data_gp_bands_fig_spacetime[1:3, 3])
cb.colorrange = (0.0, 2000.0)
cb.label = "Cell density (cells/mm²)"
save("figures/jin_assay_data_spacetime_plots.pdf", jin_assay_data_gp_bands_fig_spacetime, px_per_unit = 2)

#####################################################################
## Delay, Fisher-Kolmogorov, 12000
#####################################################################

# Extract the parameters 
x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers, gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params = set_parameters(10, assay_data[2], 2, x_scale, t_scale)

# Calibrate 
bootstrap_setup = @set bootstrap_setup.B = 20
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize)

# Perform the full simulation
T_params = [-2.0, 5.0]
D_params = [0.0050]
R_params = [1.7e-3*x_scale^2, 1.4]
lowers, uppers = [0.9, 0.9, 0.9, 0.9], [1.5, 1.5, 1.5, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 3

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params = D_params, R_params = R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the density values 
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize, delay_scales = [T_params[1], T_params[2]/t_scale], diffusion_scales = D_params*x_scale^2/t_scale, reaction_scales = R_params[2]/t_scale)
delayCurveFigure, diffusionCurveFigure, reactionCurveFigure = curve_results(bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)
uvals = LinRange(extrema(u)..., 500)/x_scale^2
pdeDataFigure = pde_results(x_pde, t_pde, u_pde, pde_data, bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)
pdeGPFigure = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)

#####################################################################
## Delay, Porous-Fisher, 12000
#####################################################################

# Extract the parameters 
x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers, gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params = set_parameters(11, assay_data[2], 2, x_scale, t_scale)

# Calibrate 
bootstrap_setup = @set bootstrap_setup.B = 5
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize)

# Perform the full simulation
T_params = [-2.0, 5.0]
D_params = [1.7e-3*x_scale^2, 0.01]
R_params = [1.7e-3*x_scale^2, 1.4]
lowers, uppers = [0.9, 0.9, 0.9, 0.9], [1.5, 1.5, 1.5, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 3
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params = D_params, R_params = R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the density values 
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize, delay_scales = [T_params[1], T_params[2]/t_scale], diffusion_scales = D_params[2]*x_scale^2/t_scale, reaction_scales = R_params[2]/t_scale)
delayCurveFigure, diffusionCurveFigure, reactionCurveFigure = curve_results(bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)
uvals = LinRange(extrema(u)..., 500)/x_scale^2
pdeDataFigure = pde_results(x_pde, t_pde, u_pde, pde_data, bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)
pdeGPFigure = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)

#####################################################################
## Delay, Affine Diffusion, 12000
#####################################################################

# Extract the parameters 
x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers, gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params = set_parameters(12, assay_data[2], 2, x_scale, t_scale)

# Calibrate 
bootstrap_setup = @set bootstrap_setup.B = 5
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
optim_setup = Optim.Options(f_reltol = 1e-4, x_reltol = 1e-4, g_reltol = 1e-4, outer_f_reltol = 1e-4, outer_x_reltol = 1e-4, outer_g_reltol = 1e-4)

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize, delay_resolution = (1200, 800), diffusion_resolution = (1200, 800))

# Perform the full simulation
T_params = [-3.0, 5.5]
D_params = [1.7e-3*x_scale^2, 0.0084792, 0.07]
R_params = [1.7e-3*x_scale^2, 1.7]
lowers, uppers = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params = D_params, R_params = R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the density values 
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize, delay_scales = [T_params[1], T_params[2]/t_scale], diffusion_scales = D_params[2]*x_scale^2/t_scale, reaction_scales = R_params[2]/t_scale, delay_resolution = (1200, 800), diffusion_resolution = (1200, 800))
delayCurveFigure, diffusionCurveFigure, reactionCurveFigure = curve_results(bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)
uvals = LinRange(extrema(u)..., 500)/x_scale^2
pdeDataFigure = pde_results(x_pde, t_pde, u_pde, pde_data, bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)
pdeGPFigure = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp; fontsize = fontsize, x_scale = x_scale, t_scale = t_scale)

