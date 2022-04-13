#####################################################################
## Load the required packages
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
using Optim                 # For optimisation
using Printf                # For @sprintf 
using KernelDensity         # For kernel density estimates

#####################################################################
## Set some global parameters 
#####################################################################

fontsize = 26
colors = [:black, :blue, :red, :magenta, :green]
alphabet = join('a':'z')
legendentries = OrderedDict("0" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[1]),
    "12" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[2]),
    "24" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[3]),
    "36" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[4]),
    "48" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[5]))
LinearAlgebra.BLAS.set_num_threads(1)

#####################################################################
## Read in the data from Jin et al. (2016)
#####################################################################

function prepare_data(filename) # https://discourse.julialang.org/t/failed-to-precompile-csv-due-to-load-error/70146/2
    data, header = readdlm(filename, ',', header=true)
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
## Figure X: Plotting the density data from Jin et al. (2016)
#####################################################################

assay_plots = Array{Axis}(undef, 3, 2)
jin_assay_data_fig = Figure(fontsize=fontsize, resolution=(1800, 1000))
plot_cart_idx = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
for (k, (i, j)) in enumerate(plot_cart_idx)
    data = assay_data[k]
    assay_plots[i, j] = Axis(jin_assay_data_fig[i, j], xlabel="Position (μm)", ylabel="Cell density\n(cells/μm²)",
        title="($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign=:left)
    for (s, T) in enumerate(unique(data.Time))
        scatter!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.Dens1[data.Time.==T] / x_scale^2, color=colors[s], markersize=3)
        scatter!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.Dens2[data.Time.==T] / x_scale^2, color=colors[s], markersize=3)
        scatter!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.Dens3[data.Time.==T] / x_scale^2, color=colors[s], markersize=3)
        lines!(assay_plots[i, j], data.Position[data.Time.==T] * x_scale, data.AvgDens[data.Time.==T] / x_scale^2, color=colors[s])
    end
    hlines!(assay_plots[i, j], K / x_scale^2, color=:black)
    ylims!(assay_plots[i, j], 0.0, 0.002)
end

Legend(jin_assay_data_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation=:horizontal, labelsize=fontsize, titlesize=fontsize, titleposition=:left)
save("figures/jin_assay_data.pdf", jin_assay_data_fig, px_per_unit=2)

#####################################################################
## Figure X: Plotting the Gaussian processes fit to the data from Jin et al. (2016).
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig = Figure(fontsize=fontsize, resolution=(1800, 1000))
gp_plots = Array{Axis}(undef, 3, 2)
plot_cart_idx = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]

for (k, (i, j)) in enumerate(plot_cart_idx)
    data = assay_data[k]
    x = repeat(data.Position, outer=3)
    t = repeat(data.Time, outer=3)
    u = vcat(data.Dens1, data.Dens2, data.Dens3)
    gp_dat = EquationLearning.fit_GP(x, t, u)
    μ, Σ = predict_f(gp_dat, [EquationLearning.scale_unit(vec(data.Position)'); EquationLearning.scale_unit(vec(data.Time)')])
    lower = μ .- 2sqrt.(Σ)
    upper = μ .+ 2sqrt.(Σ)
    gp_plots[i, j] = Axis(jin_assay_data_gp_bands_fig[i, j], xlabel="Position (μm)", ylabel="Cell density\n(cells/μm²)",
        title="($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign=:left)
    for (s, T) in enumerate(unique(t))
        scatter!(gp_plots[i, j], x[t.==T] * x_scale, u[t.==T] / x_scale^2, color=colors[s], markersize=3)
        lines!(gp_plots[i, j], data.Position[data.Time.==T] * x_scale, μ[data.Time.==T] / x_scale^2, color=colors[s])
        band!(gp_plots[i, j], data.Position[data.Time.==T] * x_scale, upper[data.Time.==T] / x_scale^2, lower[data.Time.==T] / x_scale^2, color=(colors[s], 0.35))
    end
    hlines!(gp_plots[i, j], K / x_scale^2, color=:black)
    ylims!(gp_plots[i, j], 0.0, 0.002)
end

Legend(jin_assay_data_gp_bands_fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation=:horizontal, labelsize=fontsize, titlesize=fontsize, titleposition=:left)
save("figures/jin_assay_data_gp_plots.pdf", jin_assay_data_gp_bands_fig, px_per_unit=2)

#####################################################################
## Figure X: Plotting the space-time diagram for the Gaussian process
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig_spacetime = Figure(fontsize=fontsize, resolution=(1800, 1000))
spacetime_plots = Array{Axis}(undef, 3, 2)
plot_cart_idx = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]

for (k, (i, j)) in enumerate(plot_cart_idx)
    data = assay_data[k]
    x = repeat(data.Position, outer=3)
    t = repeat(data.Time, outer=3)
    u = vcat(data.Dens1, data.Dens2, data.Dens3)
    gp_dat = EquationLearning.fit_GP(x, t, u; num_restarts=50)
    x_rng = LinRange(extrema(x)..., 200)
    t_rng = LinRange(extrema(t)..., 200)
    X_rng = repeat(x_rng, outer=length(t_rng))
    T_rng = repeat(t_rng, inner=length(x_rng))
    x̃_rng = EquationLearning.scale_unit(X_rng)
    t̃_rng = EquationLearning.scale_unit(T_rng)
    Xₛ = [vec(x̃_rng)'; vec(t̃_rng)']
    μ, _ = predict_f(gp_dat, Xₛ)
    spacetime_plots[i, j] = Axis(jin_assay_data_gp_bands_fig_spacetime[i, j], xlabel="Position (μm)", ylabel="Time (h)",
        title="($(alphabet[k])): $(10+2*(k-1)),000 cells per well",
        titlealign=:left)
    heatmap!(spacetime_plots[i, j], X_rng * x_scale, T_rng * t_scale, μ / x_scale^2; colorrange=(0.0, 0.002))
end

cb = Colorbar(jin_assay_data_gp_bands_fig_spacetime[1:3, 3], colorrange=(0, 0.002))
cb.label = "Cell density (cells/μm²)"
save("figures/jin_assay_data_spacetime_plots.pdf", jin_assay_data_gp_bands_fig_spacetime, px_per_unit=2)

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
meshPoints = LinRange(75.0 / x_scale, 1875.0 / x_scale, 500)
pde_setup = EquationLearning.PDE_Setup(meshPoints, LHS, RHS, finalTime, δt, alg)

## Setup bootstrapping 
nₓ = 50
nₜ = 50
bootₓ = LinRange(75.0 / x_scale, 1875.0 / x_scale, nₓ)
bootₜ = LinRange(0.0, 48.0 / t_scale, nₜ)
B = 100
τ = (0.0, 0.0)
Optim_Restarts = 1
constrained = false
obj_scale_GLS = log
obj_scale_PDE = log
init_weight = 10.0
show_losses = false
bootstrap_setup = EquationLearning.Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, init_weight, show_losses)

## Setup the GP parameters 
num_restarts = 250
ℓₓ = log.([1e-6, 1.0])
ℓₜ = log.([1e-6, 1.0])
nugget = 1e-5
GP_Restarts = 250

## Optimisation options 
optim_setup = Optim.Options(iterations=10, f_reltol=1e-4, x_reltol=1e-4, g_reltol=1e-4, outer_f_reltol=1e-4, outer_x_reltol=1e-4, outer_g_reltol=1e-4)

assaydata = deepcopy(assay_data)
for j = 1:6
    rename!(assay_data[j], names(assay_data[j])[1] => :Column)
    assay_data[j] = assay_data[j][assay_data[j][:, :Column].!=1.0, :]
end

#####################################################################
## Model fits
#####################################################################
fontsize = 26
function model_fits(assay_data, dat_idx,
    bootstrap_setup, GP_Restarts, nugget,
    T_params1, T_params2, T_params3, T_params4, T_params5,
    D_params1, D_params2, D_params3, D_params4, D_params5,
    R_params1, R_params2, R_params3, R_params4, R_params5,
    x_scale, t_scale,
    pde_setup, optim_setup, seed)
    # Compute GP/joint GP
    Random.seed!(seed)
    dat = assay_data[dat_idx]
    x = repeat(dat.Position, outer=3)
    t = repeat(dat.Time, outer=3)
    u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
    x_pde = dat.Position
    t_pde = dat.Time
    u_pde = dat.AvgDens
    σ = log.([1e-6, 7std(u)])
    σₙ = log.([1e-6, 7std(u)])
    gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
    gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)
    nₓ = length(bootstrap_setup.bootₓ)
    nₜ = length(bootstrap_setup.bootₜ)
    nₓnₜ = nₓ * nₜ
    zvals = randn(4nₓnₜ, bootstrap_setup.B)
    # Fit Fisher-Kolmogorov model without delay
    Random.seed!(seed + 1)
    T = (t, α, p) -> 1.0
    D = (u, β, p) -> β[1] * p[1]
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> 0.0
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params1
    D_params = D_params1
    R_params = R_params1
    α₀ = Vector{Float64}([])
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99]
    uppers = [1.01, 1.01]
    bgp1 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, zvals, verbose=false)
    # Fit Fisher-Kolmogorov model with delay
    Random.seed!(seed + 2)
    T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
    D = (u, β, p) -> β[1] * p[1]
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> 0.0
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params2
    D_params = D_params2
    R_params = R_params2
    α₀ = [1.0, 1.0]
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99, 0.99, 0.99]
    uppers = [1.01, 1.01, 1.01, 1.01]
    bgp2 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, zvals, verbose=false)
    # Fit Porous-Fisher model without delay 
    Random.seed!(seed + 3)
    T = (t, α, p) -> 1.0
    D = (u, β, p) -> β[1] * p[2] * (u / p[1])
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> β[1] * p[2] / p[1]
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params3
    D_params = D_params3
    R_params = R_params3
    α₀ = Vector{Float64}([])
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99]
    uppers = [1.01, 1.01]
    bgp3 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, zvals, verbose=false)
    # Fit a Porous-Fisher model with delay
    Random.seed!(seed + 4)
    T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
    D = (u, β, p) -> β[1] * p[2] * (u / p[1])
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> β[1] * p[2] / p[1]
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params4
    D_params = D_params4
    R_params = R_params4
    α₀ = [1.0, 1.0]
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99, 0.99, 0.99]
    uppers = [1.01, 1.01, 1.01, 1.01]
    bgp4 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, zvals, verbose=false)
    # Fit an affined generalised Porous-Fisher model 
    Random.seed!(seed + 5)
    T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
    D = (u, β, p) -> u > 0 ? β[1] * p[2] + β[2] * p[3] * (u / p[1])^(β[3] * p[4]) : β[1] * p[2]
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> u > 0 ? β[2] * p[3] * β[3] * p[4] * (u / p[1])^(β[3] * p[4] - 1) / p[1] : 0.0
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params5
    D_params = D_params5
    R_params = R_params5
    α₀ = [1.0, 1.0]
    β₀ = [1.0, 1.0, 1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    uppers = [1.01, 1.01, 1.01, 1.01, 1.01, 1.01]
    bgp5 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, zvals, verbose=false)
    # Results 
    Random.seed!(seed + 6)
    delay_scales1, diffusion_scales1, reaction_scales1 = nothing, D_params1[1] * x_scale^2 / t_scale, R_params1[2] / t_scale
    delay_scales2, diffusion_scales2, reaction_scales2 = [T_params2[1], T_params2[2] / t_scale], D_params2[1] * x_scale^2 / t_scale, R_params2[2] / t_scale
    delay_scales3, diffusion_scales3, reaction_scales3 = nothing, D_params3[2] * x_scale^2 / t_scale, R_params3[2] / t_scale
    delay_scales4, diffusion_scales4, reaction_scales4 = [T_params4[1], T_params4[2] / t_scale], D_params4[2] * x_scale^2 / t_scale, R_params4[2] / t_scale
    delay_scales5, diffusion_scales5, reaction_scales5 = [T_params5[1], T_params5[2] / t_scale], [D_params5[2] * x_scale^2 / t_scale, D_params5[3] * x_scale^2 / t_scale, D_params5[4]], R_params5[2] / t_scale
    delay_scales = [delay_scales1, delay_scales2, delay_scales3, delay_scales4, delay_scales5]
    diffusion_scales = [diffusion_scales1, diffusion_scales2, diffusion_scales3, diffusion_scales4, diffusion_scales5]
    reaction_scales = [reaction_scales1, reaction_scales2, reaction_scales3, reaction_scales4, reaction_scales5]
    res = AllResults(x_pde, t_pde, u_pde, bgp1, bgp2, bgp3, bgp4, bgp5; delay_scales, diffusion_scales, reaction_scales, x_scale, t_scale)
    return res
end

function plot_fisher_kolmogorov_delay(bgp::BootResults, x_scale, t_scale, filename, colors, dat_idx, assay_data, fontsize)
    dat = assay_data[dat_idx]
    x_pde = dat.Position
    t_pde = dat.Time 
    u_pde = dat.AvgDens
    pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType="gp")
    pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType="data")
    T_params = bgp.T_params
    D_params = bgp.D_params
    R_params = bgp.R_params
    trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level=0.05, delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=D_params[1] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
    resultFigures = Figure(fontsize=fontsize, resolution=(1800, 1000))
    delayDensityAxes = Vector{Axis}(undef, tt)
    diffusionDensityAxes = Vector{Axis}(undef, d)
    reactionDensityAxes = Vector{Axis}(undef, r)
    for i = 1:tt
        delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel=i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel="Probability density",
            title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
            titlealign=:left)
        densdat = KernelDensity.kde(trv[i, :])
        in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
        lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
        CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
        band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
    end
    for i = 1:d
        diffusionDensityAxes[i] = Axis(resultFigures[2, i], xlabel=L"$\beta_%$i$ (μm²/h)", ylabel="Probability density",
            title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+3], diffusionCIs[i, 1], diffusionCIs[i, 2]),
            titlealign=:left)
        densdat = KernelDensity.kde(dr[i, :])
        in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
        lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
        CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
        band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
    end
    for i = 1:r
        reactionDensityAxes[i] = Axis(resultFigures[2, i+1], xlabel=L"$\gamma_%$i$ (1/h)", ylabel="Probability density",
            title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+4], reactionCIs[i, 1], reactionCIs[i, 2]),
            titlealign=:left)
        densdat = kde(rr[i, :])
        in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
        lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
        CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
        band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
    end
    Tu_vals, _, _, u_vals, t_vals = curve_values(bgp; level=0.05, x_scale=x_scale, t_scale=t_scale)
    delayAxis = Axis(resultFigures[1, 3], xlabel=L"$t$ (h)", ylabel=L"$T(t)$", title="(c): Delay curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
    lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
    band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color=(:blue, 0.35))
    diffusionAxis = Axis(resultFigures[2, 3], xlabel=L"$u$ (cells/μm²)", ylabel=L"$T(t)D(u)$ (μm²/h)", title="(f): Diffusion curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
    Du_vals0 = delay_product(bgp, 0.0; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals12 = delay_product(bgp, 12.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals24 = delay_product(bgp, 24.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals36 = delay_product(bgp, 36.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals48 = delay_product(bgp, 48.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals0[1], color=colors[1])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals12[1], color=colors[2])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals24[1], color=colors[3])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals36[1], color=colors[4])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals48[1], color=colors[5])
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals0[3], Du_vals0[2], color=(colors[1], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals12[3], Du_vals12[2], color=(colors[2], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals24[3], Du_vals24[2], color=(colors[3], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals36[3], Du_vals36[2], color=(colors[4], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals48[3], Du_vals48[2], color=(colors[5], 0.1))
    reactionAxis = Axis(resultFigures[3, 3], xlabel=L"$u$ (cells/μm²)", ylabel=L"$T(t)R(u)$ (cells/μm²h)", title="(i): Reaction curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
    Ru_vals0 = delay_product(bgp, 0.0; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals12 = delay_product(bgp, 12.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals24 = delay_product(bgp, 24.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals36 = delay_product(bgp, 36.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals48 = delay_product(bgp, 48.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals0[1], color=colors[1])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals12[1], color=colors[2])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals24[1], color=colors[3])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals36[1], color=colors[4])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals48[1], color=colors[5])
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals0[3], Ru_vals0[2], color=(colors[1], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals12[3], Ru_vals12[2], color=(colors[2], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals24[3], Ru_vals24[2], color=(colors[3], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals36[3], Ru_vals36[2], color=(colors[4], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals48[3], Ru_vals48[2], color=(colors[5], 0.1))
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp)
    err_CI = error_comp(bgp, pde_data, x_pde, t_pde, u_pde)
    M = length(bgp.pde_setup.δt)
    dataAxis = Axis(resultFigures[3, 1], xlabel=L"$x$ (μm)", ylabel=L"$u(x, t)$ (cells/μm²)", title=@sprintf("(g): PDE (Spline IC)\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign=:left)
    @views for j in 1:M
        lines!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color=colors[j])
        band!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color=(colors[j], 0.35))
        CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color=colors[j], markersize=3)
    end
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
    err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
    M = length(bgp.pde_setup.δt)
    GPAxis = Axis(resultFigures[3, 2], xlabel=L"$x$ (μm)", ylabel=L"$u(x, t)$ (cells/μm²)", title=@sprintf("(h): PDE (Sampled IC)\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign=:left)
    @views for j in 1:M
        lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color=colors[j])
        band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color=(colors[j], 0.35))
        CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color=colors[j], markersize=3)
    end
    Legend(resultFigures[1, 4], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation=:vertical, labelsize=fontsize, titlesize=fontsize, titleposition=:top)
    save("figures/$filename", resultFigures, px_per_unit=2)
    return resultFigures
end

function plot_generalised_fkpp_delay(bgp::BootResults, x_scale, t_scale, filename, colors, dat_idx, assay_data, fontsize)
    dat = assay_data[dat_idx]
    x_pde = dat.Position
    t_pde = dat.Time
    u_pde = dat.AvgDens
    pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType="gp")
    T_params = bgp.T_params
    D_params = bgp.D_params
    R_params = bgp.R_params
    trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level=0.05, delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=[D_params[2] * x_scale^2 / t_scale, D_params[3] * x_scale^2 / t_scale, D_params[4]], reaction_scales=R_params[2] / t_scale)
    resultFigures = Figure(fontsize=fontsize, resolution=(1400, 1800))
    delayDensityAxes = Vector{Axis}(undef, tt)
    diffusionDensityAxes = Vector{Axis}(undef, d)
    reactionDensityAxes = Vector{Axis}(undef, r)
    for i = 1:tt
        delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel=i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel="Probability density",
            title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
            titlealign=:left)
        densdat = KernelDensity.kde(trv[i, :])
        in_range = minimum(trv[i, :]) .< densdat.x .< maximum(trv[i, :])
        lines!(delayDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
        CI_range = delayCIs[i, 1] .< densdat.x .< delayCIs[i, 2]
        band!(delayDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
    end
    for i = 1:d
        diffusionDensityAxes[i] = Axis(resultFigures[i ≠ 3 ? 2 : 3, i ≠ 3 ? i : 1], xlabel=i ≠ 3 ? L"$\beta_%$i$ (μm²/h)" : L"$\beta_%$i$", ylabel="Probability density",
            title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+2], diffusionCIs[i, 1], diffusionCIs[i, 2]),
            titlealign=:left)
        densdat = KernelDensity.kde(dr[i, :])
        in_range = minimum(dr[i, :]) .< densdat.x .< maximum(dr[i, :])
        lines!(diffusionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
        CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
        band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
    end
    for i = 1:r
        reactionDensityAxes[i] = Axis(resultFigures[3, 2], xlabel=L"$\gamma_%$i$ (1/h)", ylabel="Probability density",
            title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i+5], reactionCIs[i, 1], reactionCIs[i, 2]),
            titlealign=:left)
        densdat = kde(rr[i, :])
        in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
        lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
        CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
        band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
    end
    Tu_vals, _, _, u_vals, t_vals = curve_values(bgp; level=0.05, x_scale=x_scale, t_scale=t_scale)
    delayAxis = Axis(resultFigures[4, 1], xlabel=L"$t$ (h)", ylabel=L"$T(t)$", title="(g): Delay curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
    lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
    band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color=(:blue, 0.35))
    diffusionAxis = Axis(resultFigures[4, 2], xlabel=L"$u$ (cells/μm²)", ylabel=L"$T(t)D(u)$ (μm²/h)", title="(h): Diffusion curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
    Du_vals0 = delay_product(bgp, 0.0; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals12 = delay_product(bgp, 12.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals24 = delay_product(bgp, 24.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals36 = delay_product(bgp, 36.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    Du_vals48 = delay_product(bgp, 48.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals0[1], color=colors[1])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals12[1], color=colors[2])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals24[1], color=colors[3])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals36[1], color=colors[4])
    lines!(diffusionAxis, u_vals / x_scale^2, Du_vals48[1], color=colors[5])
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals0[3], Du_vals0[2], color=(colors[1], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals12[3], Du_vals12[2], color=(colors[2], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals24[3], Du_vals24[2], color=(colors[3], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals36[3], Du_vals36[2], color=(colors[4], 0.1))
    band!(diffusionAxis, u_vals / x_scale^2, Du_vals48[3], Du_vals48[2], color=(colors[5], 0.1))
    reactionAxis = Axis(resultFigures[5, 1], xlabel=L"$u$ (cells/μm²)", ylabel=L"$T(t)R(u)$ (cells/μm²h)", title="(i): Reaction curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
    Ru_vals0 = delay_product(bgp, 0.0; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals12 = delay_product(bgp, 12.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals24 = delay_product(bgp, 24.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals36 = delay_product(bgp, 36.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    Ru_vals48 = delay_product(bgp, 48.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals0[1], color=colors[1])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals12[1], color=colors[2])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals24[1], color=colors[3])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals36[1], color=colors[4])
    lines!(reactionAxis, u_vals / x_scale^2, Ru_vals48[1], color=colors[5])
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals0[3], Ru_vals0[2], color=(colors[1], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals12[3], Ru_vals12[2], color=(colors[2], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals24[3], Ru_vals24[2], color=(colors[3], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals36[3], Ru_vals36[2], color=(colors[4], 0.1))
    band!(reactionAxis, u_vals / x_scale^2, Ru_vals48[3], Ru_vals48[2], color=(colors[5], 0.1))
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
    err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
    M = length(bgp.pde_setup.δt)
    GPAxis = Axis(resultFigures[5, 2], xlabel=L"$x$ (μm)", ylabel=L"$u(x, t)$ (cells/μm²)", title=@sprintf("(j): PDE (Sampled IC)\nError: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign=:left)
    @views for j in 1:M
        lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color=colors[j])
        band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color=(colors[j], 0.35))
        CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color=colors[j], markersize=3)
    end
    Legend(resultFigures[1, 3], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation=:vertical, labelsize=fontsize, titlesize=fontsize, titleposition=:top)
    save("figures/$filename", resultFigures, px_per_unit=2)
    return resultFigures
end

function plot_pde_soln!(fig, bgp, i, j, alphabet, x_scale, colors, dat_idx, assay_data, K)
    dat = assay_data[dat_idx]
    x_pde = dat.Position
    t_pde = dat.Time
    u_pde = dat.AvgDens
    pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType="gp")
    soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
    err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
    M = length(bgp.pde_setup.δt)
    GPAxis = Axis(fig[i, j], xlabel=L"$x$ (μm)", ylabel=L"$u(x, t)$ (cells/μm²)", title=@sprintf("(%s): %i,000 cells per well.\nPDE Error: (%.4g, %.4g)", alphabet, 10 + 2 * (dat_idx - 1), err_CI[1], err_CI[2]), titlealign=:left)
    @views for j in 1:M
        lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color=colors[j])
        band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color=(colors[j], 0.35))
        CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color=colors[j], markersize=3)
    end
    hlines!(GPAxis, K / x_scale^2, color=:black)
    ylims!(GPAxis, 0.0, 0.002)
    return nothing
end

function plot_pde_soln(bgp1, bgp2, bgp3, bgp4, bgp5, bgp6, x_scale, colors, assay_data, fontsize, filename, K)
    fig = Figure(fontsize=fontsize, resolution=(1800, 1000))
    plot_idx = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    bgps = [bgp1, bgp2, bgp3, bgp4, bgp5, bgp6]
    alphabet = join('a':'z')
    [plot_pde_soln!(fig, bgps[i], plot_idx[i][1], plot_idx[i][2], alphabet[i], x_scale, colors, i, assay_data, K) for i in 1:6]
    Legend(fig[0, 1:2], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation=:horizontal, labelsize=fontsize, titlesize=fontsize, titleposition=:left)
    save("figures/$filename", fig, px_per_unit=2)
    return fig
end

T_params_10_1 = Vector{Float64}([])
D_params_10_1 = [180.0 * t_scale / x_scale^2]
R_params_10_1 = [K, 0.047 * t_scale]
T_params_10_2 = [-1.3, 0.2 * t_scale]
D_params_10_2 = [200.0 * t_scale / x_scale^2]
R_params_10_2 = [K, 0.055 * t_scale]
T_params_10_3 = Vector{Float64}([])
D_params_10_3 = [K, 650.0 * t_scale / x_scale^2]
R_params_10_3 = [K, 0.047 * t_scale]
T_params_10_4 = [-1.3, 0.2 * t_scale]
D_params_10_4 = [K, 300.0 * t_scale / x_scale^2]
R_params_10_4 = [K, 0.055 * t_scale]
T_params_10_5 = [-1.0, 0.2 * t_scale]
D_params_10_5 = [K, 100.0 * t_scale / x_scale^2, 4000.0 * t_scale / x_scale^2, 1.6]
R_params_10_5 = [K, 0.07 * t_scale]

T_params_12_1 = Vector{Float64}([])
D_params_12_1 = [150.0 * t_scale / x_scale^2]
R_params_12_1 = [K, 0.046 * t_scale]
T_params_12_2 = [-1.6, 0.2 * t_scale]
D_params_12_2 = [160.0 * t_scale / x_scale^2]
R_params_12_2 = [K, 0.057 * t_scale]
T_params_12_3 = Vector{Float64}([])
D_params_12_3 = [K, 500.0 * t_scale / x_scale^2]
R_params_12_3 = [K, 0.046 * t_scale]
T_params_12_4 = [-1.5, 0.2 * t_scale]
D_params_12_4 = [K, 400.0 * t_scale / x_scale^2]
R_params_12_4 = [K, 0.057 * t_scale]
T_params_12_5 = [-1.8, 0.22 * t_scale]
D_params_12_5 = [K, 350.0 * t_scale / x_scale^2, 3200.0 * t_scale / x_scale^2, 3.47]
R_params_12_5 = [K, 0.07 * t_scale]

T_params_14_1 = Vector{Float64}([])
D_params_14_1 = [600.0 * t_scale / x_scale^2]
R_params_14_1 = [K, 0.048 * t_scale]
T_params_14_2 = [-1.0, 0.25 * t_scale]
D_params_14_2 = [600.0 * t_scale / x_scale^2]
R_params_14_2 = [K, 0.056 * t_scale]
T_params_14_3 = Vector{Float64}([])
D_params_14_3 = [K, 1800.0 * t_scale / x_scale^2]
R_params_14_3 = [K, 0.048 * t_scale]
T_params_14_4 = [-0.8, 0.25 * t_scale]
D_params_14_4 = [K, 1800.0 * t_scale / x_scale^2]
R_params_14_4 = [K, 0.056 * t_scale]
T_params_14_5 = [-1.6, 0.25 * t_scale]
D_params_14_5 = [K, 482.0 * t_scale / x_scale^2, 3775.0 * t_scale / x_scale^2, 1.9]
R_params_14_5 = [K, 0.072 * t_scale]

T_params_16_1 = Vector{Float64}([])
D_params_16_1 = [530.0 * t_scale / x_scale^2]
R_params_16_1 = [K, 0.050 * t_scale]
T_params_16_2 = [-1.0, 0.2 * t_scale]
D_params_16_2 = [600.0 * t_scale / x_scale^2]
R_params_16_2 = [K, 0.06 * t_scale]
T_params_16_3 = Vector{Float64}([])
D_params_16_3 = [K, 1200.0 * t_scale / x_scale^2]
R_params_16_3 = [K, 0.049 * t_scale]
T_params_16_4 = [-1.3, 0.2 * t_scale]
D_params_16_4 = [K, 1200.0 * t_scale / x_scale^2]
R_params_16_4 = [K, 0.06 * t_scale]
T_params_16_5 = [-3.0, 0.3 * t_scale]
D_params_16_5 = [K, 604.0 * t_scale / x_scale^2, 3773.0 * t_scale / x_scale^2, 3.5]
R_params_16_5 = [K, 0.08 * t_scale]

T_params_18_1 = Vector{Float64}([])
D_params_18_1 = [600.0 * t_scale / x_scale^2]
R_params_18_1 = [K, 0.056 * t_scale]
T_params_18_2 = [-1.3, 0.2 * t_scale]
D_params_18_2 = [650.0 * t_scale / x_scale^2]
R_params_18_2 = [K, 0.069 * t_scale]
T_params_18_3 = Vector{Float64}([])
D_params_18_3 = [K, 1250.0 * t_scale / x_scale^2]
R_params_18_3 = [K, 0.056 * t_scale]
T_params_18_4 = [-1.3, 0.17 * t_scale]
D_params_18_4 = [K, 1220.0 * t_scale / x_scale^2]
R_params_18_4 = [K, 0.065 * t_scale]
T_params_18_5 = [-1.0, 0.15 * t_scale]
D_params_18_5 = [K, 800.0 * t_scale / x_scale^2, 2200.0 * t_scale / x_scale^2, 3.2]
R_params_18_5 = [K, 0.07 * t_scale]

T_params_20_1 = Vector{Float64}([])
D_params_20_1 = [620.0 * t_scale / x_scale^2]
R_params_20_1 = [K, 0.064 * t_scale]
T_params_20_2 = [-2.5, 0.3 * t_scale]
D_params_20_2 = [750.0 * t_scale / x_scale^2]
R_params_20_2 = [K, 0.09 * t_scale]
T_params_20_3 = Vector{Float64}([])
D_params_20_3 = [K, 1200.0 * t_scale / x_scale^2]
R_params_20_3 = [K, 0.075 * t_scale]
T_params_20_4 = [-2.0, 0.3 * t_scale]
D_params_20_4 = [K, 1500.0 * t_scale / x_scale^2]
R_params_20_4 = [K, 0.93 * t_scale]
T_params_20_5 = [-1.0, 0.15 * t_scale]
D_params_20_5 = [K, 675.0 * t_scale / x_scale^2, 1954.0 * t_scale / x_scale^2, 0.98]
R_params_20_5 = [K, 0.09 * t_scale]

res_10 = model_fits(assay_data, 1, bootstrap_setup, GP_Restarts, nugget,
    T_params_10_1, T_params_10_2, T_params_10_3, T_params_10_4, T_params_10_5,
    D_params_10_1, D_params_10_2, D_params_10_3, D_params_10_4, D_params_10_5,
    R_params_10_1, R_params_10_2, R_params_10_3, R_params_10_4, R_params_10_5,
    x_scale, t_scale,
    pde_setup, optim_setup, 2919211)

res_12 = model_fits(assay_data, 2, bootstrap_setup, GP_Restarts, nugget,
    T_params_12_1, T_params_12_2, T_params_12_3, T_params_12_4, T_params_12_5,
    D_params_12_1, D_params_12_2, D_params_12_3, D_params_12_4, D_params_12_5,
    R_params_12_1, R_params_12_2, R_params_12_3, R_params_12_4, R_params_12_5,
    x_scale, t_scale,
    pde_setup, optim_setup, 9998511)

res_14 = model_fits(assay_data, 3, bootstrap_setup, GP_Restarts, nugget,
    T_params_14_1, T_params_14_2, T_params_14_3, T_params_14_4, T_params_14_5,
    D_params_14_1, D_params_14_2, D_params_14_3, D_params_14_4, D_params_14_5,
    R_params_14_1, R_params_14_2, R_params_14_3, R_params_14_4, R_params_14_5,
    x_scale, t_scale,
    pde_setup, optim_setup, 64435211)

res_16 = model_fits(assay_data, 4, bootstrap_setup, 50, nugget,
    T_params_16_1, T_params_16_2, T_params_16_3, T_params_16_4, T_params_16_5,
    D_params_16_1, D_params_16_2, D_params_16_3, D_params_16_4, D_params_16_5,
    R_params_16_1, R_params_16_2, R_params_16_3, R_params_16_4, R_params_16_5,
    x_scale, t_scale,
    pde_setup, optim_setup, 323212329211)

res_18 = model_fits(assay_data, 5, bootstrap_setup, GP_Restarts, nugget,
    T_params_18_1, T_params_18_2, T_params_18_3, T_params_18_4, T_params_18_5,
    D_params_18_1, D_params_18_2, D_params_18_3, D_params_18_4, D_params_18_5,
    R_params_18_1, R_params_18_2, R_params_18_3, R_params_18_4, R_params_18_5,
    x_scale, t_scale,
    pde_setup, optim_setup, 331)

res_20 = model_fits(assay_data, 6, bootstrap_setup, GP_Restarts, 1e-4,
    T_params_20_1, T_params_20_2, T_params_20_3, T_params_20_4, T_params_20_5,
    D_params_20_1, D_params_20_2, D_params_20_3, D_params_20_4, D_params_20_5,
    R_params_20_1, R_params_20_2, R_params_20_3, R_params_20_4, R_params_20_5,
    x_scale, t_scale,
    pde_setup, optim_setup, 2923423431)

res_10_FKD = plot_fisher_kolmogorov_delay(res_10[2].bgp, x_scale, t_scale, "allplots10000.pdf", colors, 1, assay_data, fontsize)
res_10_GFKPP = plot_generalised_fkpp_delay(res_10[5].bgp, x_scale, t_scale, "lagergrenallplots10000.pdf", colors, 1, assay_data, fontsize)
res_12_FKD = plot_fisher_kolmogorov_delay(res_12[2].bgp, x_scale, t_scale, "allplots12000.pdf", colors, 2, assay_data, fontsize)
res_12_GFKPP = plot_generalised_fkpp_delay(res_12[5].bgp, x_scale, t_scale, "lagergrenallplots12000.pdf", colors, 2, assay_data, fontsize)
res_14_FKD = plot_fisher_kolmogorov_delay(res_14[2].bgp, x_scale, t_scale, "allplots14000.pdf", colors, 3, assay_data, fontsize)
res_14_GFKPP = plot_generalised_fkpp_delay(res_14[5].bgp, x_scale, t_scale, "lagergrenallplots14000.pdf", colors, 3, assay_data, fontsize)
res_16_FKD = plot_fisher_kolmogorov_delay(res_16[2].bgp, x_scale, t_scale, "allplots16000.pdf", colors, 4, assay_data, fontsize)
res_16_GFKPP = plot_generalised_fkpp_delay(res_16[5].bgp, x_scale, t_scale, "lagergrenallplots16000.pdf", colors, 4, assay_data, fontsize)
res_18_FKD = plot_fisher_kolmogorov_delay(res_18[2].bgp, x_scale, t_scale, "allplots18000.pdf", colors, 5, assay_data, fontsize)
res_18_GFKPP = plot_generalised_fkpp_delay(res_18[5].bgp, x_scale, t_scale, "lagergrenallplots18000.pdf", colors, 5, assay_data, fontsize)
res_20_FKD = plot_fisher_kolmogorov_delay(res_20[2].bgp, x_scale, t_scale, "allplots20000.pdf", colors, 6, assay_data, fontsize)
res_20_GFKPP = plot_generalised_fkpp_delay(res_20[5].bgp, x_scale, t_scale, "lagergrenallplots20000.pdf", colors, 6, assay_data, fontsize)

pde_figs = plot_pde_soln(res_10[2].bgp, res_12[2].bgp, res_14[2].bgp, res_16[2].bgp, res_18[2].bgp, res_20[2].bgp, x_scale, colors, assay_data, fontsize, "allpdeplots.pdf", K)