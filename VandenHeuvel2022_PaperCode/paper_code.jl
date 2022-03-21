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

fontsize = 14
colors = [:black, :blue, :red, :magenta, :green]
alphabet = join('a':'z')
legendentries = OrderedDict("0" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[1]),
    "12" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[2]),
    "24" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[3]),
    "36" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[4]),
    "48" => LineElement(linestyle = nothing, linewidth = 2.0, color = colors[5]))
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
    dat.Position = convert.(Float64, dat.Position)
    dat.Time = convert.(Float64, dat.Time)
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
    assay_plots[i, j] = Axis(jin_assay_data_fig[i, j], xlabel = "Position (μm)", ylabel = "Cell density\n(cells/μm²)",
        title = "($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign = :left)
    for (s, T) in enumerate(unique(data.Time))
        scatter!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.Dens1[data.Time.==T] / x_scale^2, color = colors[s], markersize = 3)
        scatter!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.Dens2[data.Time.==T] / x_scale^2, color = colors[s], markersize = 3)
        scatter!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.Dens3[data.Time.==T] / x_scale^2, color = colors[s], markersize = 3)
        lines!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.AvgDens[data.Time.==T] / x_scale^2, color = colors[s])
    end
    hlines!(assay_plots[i, j], K / x_scale^2, color = :black)
    ylims!(assay_plots[i, j], 0.0, 0.002)
end

Legend(jin_assay_data_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :horizontal, labelsize = fontsize, titlesize = fontsize, titleposition = :left)
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
    gp_plots[i, j] = Axis(jin_assay_data_gp_bands_fig[i, j], xlabel = "Position (μm)", ylabel = "Cell density\n(cells/μm²)",
        title = "($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign = :left)
    for (s, T) in enumerate(unique(t))
        scatter!(gp_plots[i, j], x[t.==T] * x_scale, u[t.==T] / x_scale^2, color = colors[s], markersize = 3)
        lines!(gp_plots[i, j], data.Position[data.Time.==T] * x_scale, μ[data.Time.==T] / x_scale^2, color = colors[s])
        band!(gp_plots[i, j], data.Position[data.Time.==T] * x_scale, upper[data.Time.==T] / x_scale^2, lower[data.Time.==T] / x_scale^2, color = (colors[s], 0.35))
    end
    hlines!(gp_plots[i, j], K / x_scale^2, color = :black)
    ylims!(gp_plots[i, j], 0.0, 0.002)
end

Legend(jin_assay_data_gp_bands_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :horizontal, labelsize = fontsize, titlesize = fontsize, titleposition = :left)
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
    spacetime_plots[i, j] = Axis(jin_assay_data_gp_bands_fig_spacetime[i, j], xlabel = "Position (μm)", ylabel = "Time (h)",
        title = "($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign = :left)
    heatmap!(spacetime_plots[i, j], X_rng * x_scale, T_rng * t_scale, μ / x_scale^2; colorrange = (0.0, 0.002))
end

cb = Colorbar(jin_assay_data_gp_bands_fig_spacetime[1:3, 3], colorrange = (0, 0.002))
cb.label = "Cell density (cells/μm²)"
save("figures/jin_assay_data_spacetime_plots.pdf", jin_assay_data_gp_bands_fig_spacetime, px_per_unit = 2)

#####################################################################
## Set some global parameters
#####################################################################

## Setup PDE variables 
δt = LinRange(0.0, 48.0 / t_scale, 5)
finalTime = 48.0 / t_scale
N = 1000
LHS = [0.0, 1.0, 0.0]
RHS = [0.0, -1.0, 0.0]
alg = Tsit5()
N_thin = 70
meshPoints = LinRange(25.0 / x_scale, 1875.0 / x_scale, 500)
pde_setup = EquationLearning.PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)

## Setup bootstrapping 
nₓ = 80
nₜ = 75
bootₓ = LinRange(25.0 / x_scale, 1875.0 / x_scale, nₓ)
bootₜ = LinRange(0.0, 48.0 / t_scale, nₜ)
B = 100
τ = (0.0, 0.0)
Optim_Restarts = 10
constrained = false
obj_scale_GLS = log
obj_scale_PDE = log
show_losses = false
bootstrap_setup = EquationLearning.Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, show_losses)

## Setup the GP parameters 
num_restarts = 50
ℓₓ = log.([1e-4, 1.0])
ℓₜ = log.([1e-4, 1.0])
nugget = 1e-5
GP_Restarts = 50

## Optimisation options 
optim_setup = Optim.Options(iterations = 10, f_reltol = 1e-4, x_reltol = 1e-4, g_reltol = 1e-4, outer_f_reltol = 1e-4, outer_x_reltol = 1e-4, outer_g_reltol = 1e-4)

#####################################################################
## Delay, Fisher-Kolmogorov, 12000
#####################################################################
Random.seed!(51021)
dat = assay_data[2]
x = repeat(dat.Position, outer = 3)
t = repeat(dat.Time, outer = 3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens

# Define the functions and parameters 
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [1.0, 1.0]
D_params = [1.0]
R_params = [K, 1.0]

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-1, 2std(u)])
σₙ = log.([1e-5, 2std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# Just some vectors for setting the number of parameters in each function 
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]

# Now do the bootstrapping. We start by seeing if we can learn the scales of the parameters so that we can re-scale for faster optimisation
lowers = [-6.0, 0.0, 0.001, 0.6]
uppers = [0.0, 5.0, 0.024, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)

# Look at the initial results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [1.0, 1 / t_scale], diffusion_scales = x_scale^2 / t_scale, reaction_scales = 1 / t_scale)
densityPDEFigures = Figure(fontsize = fontsize, resolution = (800, 400))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(densityPDEFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(densityPDEFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+2], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(densityPDEFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
save("figures/study_initial_fisher_kolmogorov_results_12000.pdf", densityPDEFigures, px_per_unit = 2)

# Now rescale and do again
T_params = [-1.0, 0.12 * t_scale]
D_params = [200.0 * t_scale / x_scale^2]
R_params = [K, 0.055 * t_scale]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1 
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
resultFigures = Figure(fontsize = fontsize, resolution = (1200, 800))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(resultFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(resultFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+4], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end

Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp; level = 0.05, x_scale = x_scale, t_scale = t_scale)
delayAxis = Axis(resultFigures[1, 3], xlabel = L"$t$ (h)", ylabel = L"$T(t)$", title = "(c): Delay curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color = (:blue, 0.35))
diffusionAxis = Axis(resultFigures[2, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)D(u)$ (μm²/h)", title = "(f): Diffusion curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Du_vals0 = delay_product(bgp, 0.0; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals12 = delay_product(bgp, 12.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals24 = delay_product(bgp, 24.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals36 = delay_product(bgp, 36.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals48 = delay_product(bgp, 48.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals0[1], color = colors[1])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals12[1], color = colors[2])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals24[1], color = colors[3])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals36[1], color = colors[4])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals48[1], color = colors[5])
band!(diffusionAxis, u_vals / x_scale^2, Du_vals0[3], Du_vals0[2], color = (colors[1], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals12[3], Du_vals12[2], color = (colors[2], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals24[3], Du_vals24[2], color = (colors[3], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals36[3], Du_vals36[2], color = (colors[4], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals48[3], Du_vals48[2], color = (colors[5], 0.1))
reactionAxis = Axis(resultFigures[3, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)R(u)$ (1/h)", title = "(i): Reaction curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Ru_vals0 = delay_product(bgp, 0.0; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals12 = delay_product(bgp, 12.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals24 = delay_product(bgp, 24.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals36 = delay_product(bgp, 36.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals48 = delay_product(bgp, 48.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals0[1], color = colors[1])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals12[1], color = colors[2])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals24[1], color = colors[3])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals36[1], color = colors[4])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals48[1], color = colors[5])
band!(reactionAxis, u_vals / x_scale^2, Ru_vals0[3], Ru_vals0[2], color = (colors[1], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals12[3], Ru_vals12[2], color = (colors[2], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals24[3], Ru_vals24[2], color = (colors[3], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals36[3], Ru_vals36[2], color = (colors[4], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals48[3], Ru_vals48[2], color = (colors[5], 0.1))

soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp)
err_CI = error_comp(bgp, pde_data, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
dataAxis = Axis(resultFigures[3, 1], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(g): PDE curves with spline ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
GPAxis = Axis(resultFigures[3, 2], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(h): PDE curves with sampled ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
Legend(resultFigures[1:3, 4], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :vertical, labelsize = fontsize, titlesize = fontsize, titleposition = :top)

save("figures/study_final_fisher_kolmogorov_results_12000.pdf", resultFigures, px_per_unit = 2)

#####################################################################
## Delay, Porous-Fisher, 12000
#####################################################################
Random.seed!(5167645021)
dat = assay_data[2]
x = repeat(dat.Position, outer = 3)
t = repeat(dat.Time, outer = 3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens

# Define the functions and parameters 
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [1.0, 1.0]
D_params = [K, 1.0]
R_params = [K, 1.0]

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-1, 2std(u)])
σₙ = log.([1e-5, 2std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# Just some vectors for setting the number of parameters in each function 
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]

# Now do the bootstrapping. We start by seeing if we can learn the scales of the parameters so that we can re-scale for faster optimisation
lowers = [-6.0, 0.0, 0.001, 0.6]
uppers = [0.0, 5.0, 0.048, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)

# Look at the initial results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [1.0, 1 / t_scale], diffusion_scales = x_scale^2 / t_scale, reaction_scales = 1 / t_scale)
densityPDEFigures = Figure(fontsize = fontsize, resolution = (800, 400))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(densityPDEFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(densityPDEFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+2], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(densityPDEFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
save("figures/study_initial_porous_fisher_results_12000.pdf", densityPDEFigures, px_per_unit = 2)

# Now rescale and do again
T_params = [-1.75, 0.2 * t_scale]
D_params = [K, 500.0 * t_scale / x_scale^2]
R_params = [K, 0.055 * t_scale]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 3 
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[2] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
resultFigures = Figure(fontsize = fontsize, resolution = (1200, 800))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(resultFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(resultFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+4], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end

Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp; level = 0.05, x_scale = x_scale, t_scale = t_scale)
delayAxis = Axis(resultFigures[1, 3], xlabel = L"$t$ (h)", ylabel = L"$T(t)$", title = "(c): Delay curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color = (:blue, 0.35))
diffusionAxis = Axis(resultFigures[2, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)D(u)$ (μm²/h)", title = "(f): Diffusion curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Du_vals0 = delay_product(bgp, 0.0; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals12 = delay_product(bgp, 12.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals24 = delay_product(bgp, 24.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals36 = delay_product(bgp, 36.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals48 = delay_product(bgp, 48.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals0[1], color = colors[1])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals12[1], color = colors[2])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals24[1], color = colors[3])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals36[1], color = colors[4])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals48[1], color = colors[5])
band!(diffusionAxis, u_vals / x_scale^2, Du_vals0[3], Du_vals0[2], color = (colors[1], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals12[3], Du_vals12[2], color = (colors[2], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals24[3], Du_vals24[2], color = (colors[3], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals36[3], Du_vals36[2], color = (colors[4], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals48[3], Du_vals48[2], color = (colors[5], 0.1))
reactionAxis = Axis(resultFigures[3, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)R(u)$ (1/h)", title = "(i): Reaction curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Ru_vals0 = delay_product(bgp, 0.0; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals12 = delay_product(bgp, 12.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals24 = delay_product(bgp, 24.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals36 = delay_product(bgp, 36.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals48 = delay_product(bgp, 48.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals0[1], color = colors[1])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals12[1], color = colors[2])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals24[1], color = colors[3])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals36[1], color = colors[4])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals48[1], color = colors[5])
band!(reactionAxis, u_vals / x_scale^2, Ru_vals0[3], Ru_vals0[2], color = (colors[1], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals12[3], Ru_vals12[2], color = (colors[2], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals24[3], Ru_vals24[2], color = (colors[3], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals36[3], Ru_vals36[2], color = (colors[4], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals48[3], Ru_vals48[2], color = (colors[5], 0.1))

soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp)
err_CI = error_comp(bgp, pde_data, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
dataAxis = Axis(resultFigures[3, 1], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(g): PDE curves with spline ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
GPAxis = Axis(resultFigures[3, 2], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(h): PDE curves with sampled ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
Legend(resultFigures[1:3, 4], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :vertical, labelsize = fontsize, titlesize = fontsize, titleposition = :top)

save("figures/study_final_porous_fisher_results_12000.pdf", resultFigures, px_per_unit = 2)

#####################################################################
## Delay, Fisher-Kolmogorov, 20000
#####################################################################
Random.seed!(51021)
dat = assay_data[6]
x = repeat(dat.Position, outer = 3)
t = repeat(dat.Time, outer = 3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens

# Define the functions and parameters 
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [1.0, 1.0]
D_params = [1.0]
R_params = [K, 1.0]

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-1, 2std(u)])
σₙ = log.([1e-5, 2std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# Just some vectors for setting the number of parameters in each function 
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]

# Now do the bootstrapping. We start by seeing if we can learn the scales of the parameters so that we can re-scale for faster optimisation
lowers = [-6.0, 0.0, 0.01, 0.6]
uppers = [0.0, 5.0, 0.048, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)

# Look at the initial results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [1.0, 1 / t_scale], diffusion_scales = x_scale^2 / t_scale, reaction_scales = 1 / t_scale)
densityPDEFigures = Figure(fontsize = fontsize, resolution = (800, 400))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(densityPDEFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(densityPDEFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+2], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(densityPDEFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
save("figures/study_initial_fisher_kolmogorov_results_20000.pdf", densityPDEFigures, px_per_unit = 2)

# Now rescale and do again
T_params = [-1.0, 0.12 * t_scale]
D_params = [200.0 * t_scale / x_scale^2]
R_params = [K, 0.055 * t_scale]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1 
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
resultFigures = Figure(fontsize = fontsize, resolution = (1200, 800))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(resultFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(resultFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+4], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end

Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp; level = 0.05, x_scale = x_scale, t_scale = t_scale)
delayAxis = Axis(resultFigures[1, 3], xlabel = L"$t$ (h)", ylabel = L"$T(t)$", title = "(c): Delay curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color = (:blue, 0.35))
diffusionAxis = Axis(resultFigures[2, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)D(u)$ (μm²/h)", title = "(f): Diffusion curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Du_vals0 = delay_product(bgp, 0.0; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals12 = delay_product(bgp, 12.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals24 = delay_product(bgp, 24.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals36 = delay_product(bgp, 36.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals48 = delay_product(bgp, 48.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals0[1], color = colors[1])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals12[1], color = colors[2])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals24[1], color = colors[3])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals36[1], color = colors[4])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals48[1], color = colors[5])
band!(diffusionAxis, u_vals / x_scale^2, Du_vals0[3], Du_vals0[2], color = (colors[1], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals12[3], Du_vals12[2], color = (colors[2], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals24[3], Du_vals24[2], color = (colors[3], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals36[3], Du_vals36[2], color = (colors[4], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals48[3], Du_vals48[2], color = (colors[5], 0.1))
reactionAxis = Axis(resultFigures[3, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)R(u)$ (1/h)", title = "(i): Reaction curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Ru_vals0 = delay_product(bgp, 0.0; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals12 = delay_product(bgp, 12.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals24 = delay_product(bgp, 24.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals36 = delay_product(bgp, 36.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals48 = delay_product(bgp, 48.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals0[1], color = colors[1])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals12[1], color = colors[2])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals24[1], color = colors[3])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals36[1], color = colors[4])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals48[1], color = colors[5])
band!(reactionAxis, u_vals / x_scale^2, Ru_vals0[3], Ru_vals0[2], color = (colors[1], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals12[3], Ru_vals12[2], color = (colors[2], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals24[3], Ru_vals24[2], color = (colors[3], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals36[3], Ru_vals36[2], color = (colors[4], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals48[3], Ru_vals48[2], color = (colors[5], 0.1))

soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp)
err_CI = error_comp(bgp, pde_data, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
dataAxis = Axis(resultFigures[3, 1], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(g): PDE curves with spline ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
GPAxis = Axis(resultFigures[3, 2], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(h): PDE curves with sampled ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
Legend(resultFigures[1:3, 4], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :vertical, labelsize = fontsize, titlesize = fontsize, titleposition = :top)

save("figures/study_final_fisher_kolmogorov_results_20000.pdf", resultFigures, px_per_unit = 2)

#####################################################################
## Delay, Porous-Fisher, 20000
#####################################################################
Random.seed!(5167645055121)
dat = assay_data[6]
x = repeat(dat.Position, outer = 3)
t = repeat(dat.Time, outer = 3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens

# Define the functions and parameters 
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [1.0, 1.0]
D_params = [K, 1.0]
R_params = [K, 1.0]

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-1, 2std(u)])
σₙ = log.([1e-5, 2std(u)])
GP_Restarts = 250
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# Just some vectors for setting the number of parameters in each function 
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]

# Now do the bootstrapping. We start by seeing if we can learn the scales of the parameters so that we can re-scale for faster optimisation
lowers = [-6.0, 0.0, 0.001, 0.6]
uppers = [0.0, 5.0, 0.048, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)

# Look at the initial results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [1.0, 1 / t_scale], diffusion_scales = x_scale^2 / t_scale, reaction_scales = 1 / t_scale)
densityPDEFigures = Figure(fontsize = fontsize, resolution = (800, 400))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(densityPDEFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(densityPDEFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+2], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(densityPDEFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
save("figures/study_initial_porous_fisher_results_20000.pdf", densityPDEFigures, px_per_unit = 2)

# Now rescale and do again
T_params = [-1.8, 0.2 * t_scale]
D_params = [K, 1500.0 * t_scale / x_scale^2]
R_params = [K, 0.1 * t_scale]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 3 
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[2] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
resultFigures = Figure(fontsize = fontsize, resolution = (1200, 800))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel = i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(trv[i, :])
    in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
    lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
    band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:d
    diffusionDensityAxes[i] = Axis(resultFigures[2, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(resultFigures[2, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+4], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end

Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp; level = 0.05, x_scale = x_scale, t_scale = t_scale)
delayAxis = Axis(resultFigures[1, 3], xlabel = L"$t$ (h)", ylabel = L"$T(t)$", title = "(c): Delay curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color = (:blue, 0.35))
diffusionAxis = Axis(resultFigures[2, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)D(u)$ (μm²/h)", title = "(f): Diffusion curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Du_vals0 = delay_product(bgp, 0.0; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals12 = delay_product(bgp, 12.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals24 = delay_product(bgp, 24.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals36 = delay_product(bgp, 36.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
Du_vals48 = delay_product(bgp, 48.0 / t_scale; type = "diffusion", x_scale = x_scale, t_scale = t_scale)
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals0[1], color = colors[1])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals12[1], color = colors[2])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals24[1], color = colors[3])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals36[1], color = colors[4])
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals48[1], color = colors[5])
band!(diffusionAxis, u_vals / x_scale^2, Du_vals0[3], Du_vals0[2], color = (colors[1], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals12[3], Du_vals12[2], color = (colors[2], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals24[3], Du_vals24[2], color = (colors[3], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals36[3], Du_vals36[2], color = (colors[4], 0.1))
band!(diffusionAxis, u_vals / x_scale^2, Du_vals48[3], Du_vals48[2], color = (colors[5], 0.1))
reactionAxis = Axis(resultFigures[3, 3], xlabel = L"$u$ (cells/μm²)", ylabel = L"$T(t)R(u)$ (1/h)", title = "(i): Reaction curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
Ru_vals0 = delay_product(bgp, 0.0; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals12 = delay_product(bgp, 12.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals24 = delay_product(bgp, 24.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals36 = delay_product(bgp, 36.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
Ru_vals48 = delay_product(bgp, 48.0 / t_scale; type = "reaction", x_scale = x_scale, t_scale = t_scale)
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals0[1], color = colors[1])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals12[1], color = colors[2])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals24[1], color = colors[3])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals36[1], color = colors[4])
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals48[1], color = colors[5])
band!(reactionAxis, u_vals / x_scale^2, Ru_vals0[3], Ru_vals0[2], color = (colors[1], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals12[3], Ru_vals12[2], color = (colors[2], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals24[3], Ru_vals24[2], color = (colors[3], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals36[3], Ru_vals36[2], color = (colors[4], 0.1))
band!(reactionAxis, u_vals / x_scale^2, Ru_vals48[3], Ru_vals48[2], color = (colors[5], 0.1))

soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp)
err_CI = error_comp(bgp, pde_data, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
dataAxis = Axis(resultFigures[3, 1], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(g): PDE curves with spline ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
GPAxis = Axis(resultFigures[3, 2], xlabel = L"$x$ (μm)", ylabel = L"$u(x, t)$ (cells/μm²)", title = @sprintf("(h): PDE curves with sampled ICs\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
Legend(resultFigures[1:3, 4], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :vertical, labelsize = fontsize, titlesize = fontsize, titleposition = :top)

save("figures/study_final_porous_fisher_results_20000.pdf", resultFigures, px_per_unit = 2)
