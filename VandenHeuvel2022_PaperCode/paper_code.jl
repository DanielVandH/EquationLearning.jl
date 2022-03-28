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
num_restarts = 250
ℓₓ = log.([1e-6, 1.0])
ℓₜ = log.([1e-6, 1.0])
nugget = 1e-5
GP_Restarts = 250

## Optimisation options 
optim_setup = Optim.Options(iterations = 10, f_reltol = 1e-4, x_reltol = 1e-4, g_reltol = 1e-4, outer_f_reltol = 1e-4, outer_x_reltol = 1e-4, outer_g_reltol = 1e-4)

#####################################################################
## 10,000 cells per well
#####################################################################
Random.seed!(5102331)
dat = assay_data[1]
x = repeat(dat.Position, outer = 3)
t = repeat(dat.Time, outer = 3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens
σ = log.([1e-6, 5std(u)])
σₙ = log.([1e-6, 5std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# First, try a Fisher-Kolmogorov model without delay.
Random.seed!(510782331)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = Vector{Float64}([])
D_params = [310.0 * t_scale / x_scale^2]
R_params = [K, 0.044 * t_scale]
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99]
uppers = [1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp1 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType = "gp")
delay1, diffusion1, reaction1 = density_results(bgp1; delay_scales = nothing, diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve1, diffusioncurve1, reactioncurve1 = curve_results(bgp1; x_scale, t_scale)
pdeplot1 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp1; x_scale, t_scale)

# Now add a Fisher-Kolmogorov model with delay.
Random.seed!(5103453457781)
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.0, 0.2 * t_scale]
D_params = [620.0 * t_scale / x_scale^2]
R_params = [K, 0.088 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp2 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp2, x_pde, t_pde, u_pde; ICType = "gp")
delay2, diffusion2, reaction2 = density_results(bgp2; delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve2, diffusioncurve2, reactioncurve2 = curve_results(bgp2; x_scale, t_scale)
pdeplot2 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp2; x_scale, t_scale)

# Now use a Porous-Fisher model without delay
Random.seed!(5888831)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = Vector{Float64}([])
D_params = [K, 1800.0 * t_scale / x_scale^2]
R_params = [K, 0.044 * t_scale]
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99]
uppers = [1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp3 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp3, x_pde, t_pde, u_pde; ICType = "gp")
delay3, diffusion3, reaction3 = density_results(bgp3; delay_scales = nothing, diffusion_scales = D_params[2] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve3, diffusioncurve3, reactioncurve3 = curve_results(bgp3; x_scale, t_scale)
pdeplot3 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp3; x_scale, t_scale)

# Use a Porous-Fisher model with delay
Random.seed!(566599931)
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.0, 0.2 * t_scale]
D_params = [K, 3200.0 * t_scale / x_scale^2]
R_params = [K, 0.070 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp4 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp4, x_pde, t_pde, u_pde; ICType = "gp")
delay4, diffusion4, reaction4 = density_results(bgp4; delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[2] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve4, diffusioncurve4, reactioncurve4 = curve_results(bgp4; x_scale, t_scale)
pdeplot4 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp4; x_scale, t_scale)

# Affine generalised Porous-Fisher model
Random.seed!(12322221)
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> u > 0 ? β[1] * p[2] + β[2] * p[3] * (u / p[1]) ^ (β[3] * p[4]) : β[1] * p[2]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> u > 0 ? β[2] * p[3] * β[3] * p[4] * (u / p[1]) ^ (β[3] * p[4] - 1) : 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.0, 0.2 * t_scale]
D_params = [K, 100.0 * t_scale / x_scale^2, 4000.0 * t_scale / x_scale^2, 1.5]
R_params = [K, 0.05 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0, 1.0, 1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01, 1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp5 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp5, x_pde, t_pde, u_pde; ICType = "gp")
delay5, diffusion5, reaction5 = density_results(bgp5; delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = [D_params[2] * x_scale^2 / t_scale, D_params[3] * x_scale^2 / t_scale, D_params[4]], reaction_scales = R_params[2] / t_scale)
delaycurve5, diffusioncurve5, reactioncurve5 = curve_results(bgp5; x_scale, t_scale)
pdeplot5 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp5; x_scale, t_scale)

# Model comparison and all results
Random.seed!(54430031)

delay_scales1, diffusion_scales1, reaction_scales1 = nothing, bgp1.D_params[1] * x_scale^2 / t_scale, bgp1.R_params[2] / t_scale
delay_scales2, diffusion_scales2, reaction_scales2 = [bgp2.T_params[1], bgp2.T_params[2] / t_scale], bgp2.D_params[1] * x_scale^2 / t_scale, bgp2.R_params[2] / t_scale
delay_scales3, diffusion_scales3, reaction_scales3 = nothing, bgp3.D_params[2] * x_scale^2 / t_scale, bgp3.R_params[2] / t_scale
delay_scales4, diffusion_scales4, reaction_scales4 = [bgp4.T_params[1], bgp4.T_params[2] / t_scale], bgp4.D_params[2] * x_scale^2 / t_scale, bgp4.R_params[2] / t_scale
delay_scales5, diffusion_scales5, reaction_scales5 = [bgp5.T_params[1], bgp5.T_params[2] / t_scale], [bgp5.D_params[2] * x_scale^2 / t_scale, bgp5.D_params[3] * x_scale^2 / t_scale, bgp5.D_params[4]], bgp5.R_params[2] / t_scale
delay_scales = [delay_scales1, delay_scales2, delay_scales3, delay_scales4, delay_scales5]
diffusion_scales = [diffusion_scales1, diffusion_scales2, diffusion_scales3, diffusion_scales4, diffusion_scales5]
reaction_scales = [reaction_scales1, reaction_scales2, reaction_scales3, reaction_scales4, reaction_scales5]

res = AllResults(x_pde, t_pde, u_pde, bgp1, bgp2, bgp3, bgp4, bgp5; delay_scales, diffusion_scales, reaction_scales, x_scale, t_scale)

#####################################################################
## 12,000 cells per well
#####################################################################
Random.seed!(5102323242331)
dat = assay_data[2]
x = repeat(dat.Position, outer = 3)
t = repeat(dat.Time, outer = 3)
u = vcat(dat.Dens1, dat.Dens2, dat.Dens3)
x_pde = dat.Position
t_pde = dat.Time
u_pde = dat.AvgDens
σ = log.([1e-6, 5std(u)])
σₙ = log.([1e-6, 5std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# First, try a Fisher-Kolmogorov model without delay.
Random.seed!(5151232331)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = Vector{Float64}([])
D_params = [250.0 * t_scale / x_scale^2]
R_params = [K, 0.044 * t_scale]
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99]
uppers = [1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp1 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType = "gp")
delay1, diffusion1, reaction1 = density_results(bgp1; delay_scales = nothing, diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve1, diffusioncurve1, reactioncurve1 = curve_results(bgp1; x_scale, t_scale)
pdeplot1 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp1; x_scale, t_scale)

# Now add a Fisher-Kolmogorov model with delay.
Random.seed!(510645477581)
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.0, 0.2 * t_scale]
D_params = [500.0 * t_scale / x_scale^2]
R_params = [K, 0.088 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp2 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp2, x_pde, t_pde, u_pde; ICType = "gp")
delay2, diffusion2, reaction2 = density_results(bgp2; delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve2, diffusioncurve2, reactioncurve2 = curve_results(bgp2; x_scale, t_scale)
pdeplot2 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp2; x_scale, t_scale)

# Now use a Porous-Fisher model without delay
Random.seed!(5888866323246431)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = Vector{Float64}([])
D_params = [K, 1300.0 * t_scale / x_scale^2]
R_params = [K, 0.043 * t_scale]
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99]
uppers = [1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp3 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp3, x_pde, t_pde, u_pde; ICType = "gp")
delay3, diffusion3, reaction3 = density_results(bgp3; delay_scales = nothing, diffusion_scales = D_params[2] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve3, diffusioncurve3, reactioncurve3 = curve_results(bgp3; x_scale, t_scale)
pdeplot3 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp3; x_scale, t_scale)

# Use a Porous-Fisher model with delay
Random.seed!(566599931)
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.0, 0.2 * t_scale]
D_params = [K, 2600.0 * t_scale / x_scale^2]
R_params = [K, 0.080 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp4 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp4, x_pde, t_pde, u_pde; ICType = "gp")
delay4, diffusion4, reaction4 = density_results(bgp4; delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = D_params[2] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
delaycurve4, diffusioncurve4, reactioncurve4 = curve_results(bgp4; x_scale, t_scale)
pdeplot4 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp4; x_scale, t_scale)

# Affine generalised Porous-Fisher model
Random.seed!(12322221)
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> u > 0 ? β[1] * p[2] + β[2] * p[3] * (u / p[1]) ^ (β[3] * p[4]) : β[1] * p[2]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
D′ = (u, β, p) -> u > 0 ? β[2] * p[3] * β[3] * p[4] * (u / p[1]) ^ (β[3] * p[4] - 1) : 0.0
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.0, 0.2 * t_scale]
D_params = [K, 100.0 * t_scale / x_scale^2, 4000.0 * t_scale / x_scale^2, 1.5]
R_params = [K, 0.05 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0, 1.0, 1.0]
γ₀ = [1.0]
lowers = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01, 1.01, 1.01]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1
bgp5 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_gp = boot_pde_solve(bgp5, x_pde, t_pde, u_pde; ICType = "gp")
delay5, diffusion5, reaction5 = density_results(bgp5; delay_scales = [T_params[1], T_params[2] / t_scale], diffusion_scales = [D_params[2] * x_scale^2 / t_scale, D_params[3] * x_scale^2 / t_scale, D_params[4]], reaction_scales = R_params[2] / t_scale)
delaycurve5, diffusioncurve5, reactioncurve5 = curve_results(bgp5; x_scale, t_scale)
pdeplot5 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp5; x_scale, t_scale)

# Model comparison and all results
Random.seed!(54430031)

delay_scales1, diffusion_scales1, reaction_scales1 = nothing, bgp1.D_params[1] * x_scale^2 / t_scale, bgp1.R_params[2] / t_scale
delay_scales2, diffusion_scales2, reaction_scales2 = [bgp2.T_params[1], bgp2.T_params[2] / t_scale], bgp2.D_params[1] * x_scale^2 / t_scale, bgp2.R_params[2] / t_scale
delay_scales3, diffusion_scales3, reaction_scales3 = nothing, bgp3.D_params[2] * x_scale^2 / t_scale, bgp3.R_params[2] / t_scale
delay_scales4, diffusion_scales4, reaction_scales4 = [bgp4.T_params[1], bgp4.T_params[2] / t_scale], bgp4.D_params[2] * x_scale^2 / t_scale, bgp4.R_params[2] / t_scale
delay_scales5, diffusion_scales5, reaction_scales5 = [bgp5.T_params[1], bgp5.T_params[2] / t_scale], [bgp5.D_params[2] * x_scale^2 / t_scale, bgp5.D_params[3] * x_scale^2 / t_scale, bgp5.D_params[4]], bgp5.R_params[2] / t_scale
delay_scales = [delay_scales1, delay_scales2, delay_scales3, delay_scales4, delay_scales5]
diffusion_scales = [diffusion_scales1, diffusion_scales2, diffusion_scales3, diffusion_scales4, diffusion_scales5]
reaction_scales = [reaction_scales1, reaction_scales2, reaction_scales3, reaction_scales4, reaction_scales5]

res = AllResults(x_pde, t_pde, u_pde, bgp1, bgp2, bgp3, bgp4, bgp5; delay_scales, diffusion_scales, reaction_scales, x_scale, t_scale)
