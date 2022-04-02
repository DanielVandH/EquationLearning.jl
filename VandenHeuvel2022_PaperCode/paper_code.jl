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
using Optim                 # For optimisation

#####################################################################
## Set some global parameters 
#####################################################################

fontsize = 14
colors = [:black, :blue, :red, :magenta, :green]
alphabet = join('a':'z')
legendentries = OrderedDict("0" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[1]),
    "12" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[2]),
    "24" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[3]),
    "36" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[4]),
    "48" => LineElement(linestyle=nothing, linewidth=2.0, color=colors[5]))
LinearAlgebra.BLAS.set_num_threads(1)

#####################################################################
## Read in the data from Jin et al. (2016).
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
## Figure X: Plotting the density data from Jin et al. (2016).
#####################################################################

assay_plots = Array{Axis}(undef, 3, 2)
jin_assay_data_fig = Figure(fontsize=fontsize)

for (k, (i, j)) in enumerate(Tuple.(CartesianIndices(assay_plots)))
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

jin_assay_data_gp_bands_fig = Figure(fontsize=fontsize)
gp_plots = Array{Axis}(undef, 3, 2)

for (k, (i, j)) in enumerate(Tuple.(CartesianIndices(gp_plots)))
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
## Figure X: Plotting the space-time diagram for the Gaussian process.
#####################################################################
Random.seed!(12991)

jin_assay_data_gp_bands_fig_spacetime = Figure(fontsize=fontsize)
spacetime_plots = Array{Axis}(undef, 3, 2)

for (k, (i, j)) in enumerate(Tuple.(CartesianIndices(spacetime_plots)))
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
nₓ = 80
nₜ = 75
bootₓ = LinRange(75.0 / x_scale, 1875.0 / x_scale, nₓ)
bootₜ = LinRange(0.0, 48.0 / t_scale, nₜ)
B = 100
τ = (0.0, 0.0)
Optim_Restarts = 2
constrained = false
obj_scale_GLS = log
obj_scale_PDE = log
show_losses = false
bootstrap_setup = EquationLearning.Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, show_losses)

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
    assay_data[j] = assay_data[j][assay_data[j][:, :Column] .!= 1.0, :]
end

#####################################################################
## Model fits
#####################################################################
function model_fits(assay_data, dat_idx,
    bootstrap_setup, GP_Restarts, nugget,
    T_params1, T_params2, T_params3, T_params4, T_params5, T_params6,
    D_params1, D_params2, D_params3, D_params4, D_params5, D_params6,
    R_params1, R_params2, R_params3, R_params4, R_params5, R_params6,
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
    bgp1 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
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
    bgp2 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
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
    bgp3 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
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
    bgp4 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
    # Fit an affined generalised Porous-Fisher model 
    Random.seed!(seed + 5)
    T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
    D = (u, β, p) -> u > 0 ? β[1] * p[2] + β[2] * p[3] * (u / p[1])^(β[3] * p[4]) : β[1] * p[2]
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> u > 0 ? β[2] * p[3] * β[3] * p[4] * (u / p[1])^(β[3] * p[4] - 1) : 0.0
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params5
    D_params = D_params5
    R_params = R_params5
    α₀ = [1.0, 1.0]
    β₀ = [1.0, 1.0, 1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    uppers = [1.01, 1.01, 1.01, 1.01, 1.01, 1.01]
    bgp5 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
    # Cloglog Fisher-Kolmogorov 
    T = (t, α, p) -> 1.0 - exp(-exp(α[1] * p[1] + α[2] * p[2] * t))
    D = (u, β, p) -> β[1] * p[1]
    R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
    D′ = (u, β, p) -> 0.0
    R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
    T_params = T_params6
    D_params = D_params6
    R_params = R_params6
    α₀ = [1.0, 1.0]
    β₀ = [1.0]
    γ₀ = [1.0]
    lowers = [0.99, 0.99, 0.99, 0.99]
    uppers = [1.01, 1.01, 1.01, 1.01]
    bgp6 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
    # Results 
    Random.seed!(seed + 6)
    delay_scales1, diffusion_scales1, reaction_scales1 = nothing, D_params1[1] * x_scale^2 / t_scale, R_params1[2] / t_scale
    delay_scales2, diffusion_scales2, reaction_scales2 = [T_params2[1], T_params2[2] / t_scale], D_params2[1] * x_scale^2 / t_scale, R_params2[2] / t_scale
    delay_scales3, diffusion_scales3, reaction_scales3 = nothing, D_params3[2] * x_scale^2 / t_scale, R_params3[2] / t_scale
    delay_scales4, diffusion_scales4, reaction_scales4 = [T_params4[1], T_params4[2] / t_scale], D_params4[2] * x_scale^2 / t_scale, R_params4[2] / t_scale
    delay_scales5, diffusion_scales5, reaction_scales5 = [T_params5[1], T_params5[2] / t_scale], [D_params5[2] * x_scale^2 / t_scale, D_params5[3] * x_scale^2 / t_scale, D_params5[4]], R_params5[2] / t_scale
    delay_scales6, diffusion_scales6, reaction_scales6 = [T_params6[1], T_params6[2] / t_scale], D_params6[1] * x_scale^2 / t_scale, R_params6[2] / t_scale
    delay_scales = [delay_scales1, delay_scales2, delay_scales3, delay_scales4, delay_scales5, delay_scales6]
    diffusion_scales = [diffusion_scales1, diffusion_scales2, diffusion_scales3, diffusion_scales4, diffusion_scales5, diffusion_scales6]
    reaction_scales = [reaction_scales1, reaction_scales2, reaction_scales3, reaction_scales4, reaction_scales5, reaction_scales6]
    res = AllResults(x_pde, t_pde, u_pde, bgp1, bgp2, bgp3, bgp4, bgp5, bgp6; delay_scales, diffusion_scales, reaction_scales, x_scale, t_scale)
    return res
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
D_params_10_5 = [K, 100.0 * t_scale / x_scale^2, 3.0 * t_scale / x_scale^2, 2.0]
R_params_10_5 = [K, 0.07 * t_scale]
T_params_10_6 = [-1.3, 0.2 * t_scale]
D_params_10_6 = [200.0 * t_scale / x_scale^2]
R_params_10_6 = [K, 0.055 * t_scale]

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
D_params_12_5 = [K, 250.0 * t_scale / x_scale^2, 1.0 * t_scale / x_scale^2, 3.0]
R_params_12_5 = [K, 0.07 * t_scale]
T_params_12_6 = [-1.6, 0.2 * t_scale]
D_params_12_6 = [160.0 * t_scale / x_scale^2]
R_params_12_6 = [K, 0.057 * t_scale]

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
D_params_14_5 = [K, 520.0 * t_scale / x_scale^2, 1.0 * t_scale / x_scale^2, 2.0]
R_params_14_5 = [K, 0.072 * t_scale]
T_params_14_6 = [-1.0, 0.25 * t_scale]
D_params_14_6 = [600.0 * t_scale / x_scale^2]
R_params_14_6 = [K, 0.056 * t_scale]

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
D_params_16_5 = [K, 550.0 * t_scale / x_scale^2, 500.0 * t_scale / x_scale^2, 3.5]
R_params_16_5 = [K, 0.08 * t_scale]
T_params_16_6 = [-1.0, 0.2 * t_scale]
D_params_16_6 = [600.0 * t_scale / x_scale^2]
R_params_16_6 = [K, 0.06 * t_scale]

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
D_params_18_5 = [K, 780.0 * t_scale / x_scale^2, 1.0 * t_scale / x_scale^2, 3.2]
R_params_18_5 = [K, 0.07 * t_scale]
T_params_18_6 = [-1.3, 0.2 * t_scale]
D_params_18_6 = [650.0 * t_scale / x_scale^2]
R_params_18_6 = [K, 0.069 * t_scale]

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
D_params_20_5 = [K, 630.0 * t_scale / x_scale^2, 1.0 * t_scale / x_scale^2, 0.98]
R_params_20_5 = [K, 0.09 * t_scale]
T_params_20_6 = [-2.5, 0.3 * t_scale]
D_params_20_6 = [750.0 * t_scale / x_scale^2]
R_params_20_6 = [K, 0.09 * t_scale]

res_10 = model_fits(assay_data, 1, bootstrap_setup, GP_Restarts, nugget,
    T_params_10_1, T_params_10_2, T_params_10_3, T_params_10_4, T_params_10_5, T_params_10_6,
    D_params_10_1, D_params_10_2, D_params_10_3, D_params_10_4, D_params_10_5, D_params_10_6,
    R_params_10_1, R_params_10_2, R_params_10_3, R_params_10_4, R_params_10_5, R_params_10_6,
    x_scale, t_scale,
    pde_setup, optim_setup, 2919211)

res_12 = model_fits(assay_data, 2, bootstrap_setup, GP_Restarts, nugget,
    T_params_12_1, T_params_12_2, T_params_12_3, T_params_12_4, T_params_12_5, T_params_12_6,
    D_params_12_1, D_params_12_2, D_params_12_3, D_params_12_4, D_params_12_5, D_params_12_6,
    R_params_12_1, R_params_12_2, R_params_12_3, R_params_12_4, R_params_12_5, R_params_12_6,
    x_scale, t_scale,
    pde_setup, optim_setup, 9998511)

res_14 = model_fits(assay_data, 3, bootstrap_setup, GP_Restarts, nugget,
    T_params_14_1, T_params_14_2, T_params_14_3, T_params_14_4, T_params_14_5, T_params_14_6,
    D_params_14_1, D_params_14_2, D_params_14_3, D_params_14_4, D_params_14_5, D_params_14_6,
    R_params_14_1, R_params_14_2, R_params_14_3, R_params_14_4, R_params_14_5, R_params_14_6,
    x_scale, t_scale,
    pde_setup, optim_setup, 64435211)

res_16 = model_fits(assay_data, 4, bootstrap_setup, GP_Restarts, nugget,
    T_params_16_1, T_params_16_2, T_params_16_3, T_params_16_4, T_params_16_5, T_params_16_6,
    D_params_16_1, D_params_16_2, D_params_16_3, D_params_16_4, D_params_16_5, D_params_16_6,
    R_params_16_1, R_params_16_2, R_params_16_3, R_params_16_4, R_params_16_5, R_params_16_6,
    x_scale, t_scale,
    pde_setup, optim_setup, 323212329211)

res_18 = model_fits(assay_data, 5, bootstrap_setup, GP_Restarts, nugget,
    T_params_18_1, T_params_18_2, T_params_18_3, T_params_18_4, T_params_18_5, T_params_18_6,
    D_params_18_1, D_params_18_2, D_params_18_3, D_params_18_4, D_params_18_5, D_params_18_6,
    R_params_18_1, R_params_18_2, R_params_18_3, R_params_18_4, R_params_18_5, R_params_18_6,
    x_scale, t_scale,
    pde_setup, optim_setup, 331)

res_20 = model_fits(assay_data, 6, bootstrap_setup, GP_Restarts, 1e-4,
    T_params_20_1, T_params_20_2, T_params_20_3, T_params_20_4, T_params_20_5, T_params_20_6,
    D_params_20_1, D_params_20_2, D_params_20_3, D_params_20_4, D_params_20_5, D_params_20_6,
    R_params_20_1, R_params_20_2, R_params_20_3, R_params_20_4, R_params_20_5, R_params_20_6,
    x_scale, t_scale,
    pde_setup, optim_setup, 2923423431)

save("figures/pdeplot10000.pdf", res_10[2].pde_plot, px_per_unit=2)
save("figures/pdeplot12000.pdf", res_12[2].pde_plot, px_per_unit=2)
save("figures/pdeplot14000.pdf", res_14[2].pde_plot, px_per_unit=2)
save("figures/pdeplot16000.pdf", res_16[2].pde_plot, px_per_unit=2)
save("figures/pdeplot18000.pdf", res_18[2].pde_plot, px_per_unit=2)
save("figures/pdeplot20000.pdf", res_20[2].pde_plot, px_per_unit=2)

