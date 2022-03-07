#####################################################################
## Load the required package
#####################################################################

using EquationLearning      # Load our actual package 
using DelimitedFiles        # For loading the density data of Jin et al. (2016).
using DataFrames            # For conveniently representing the data
using CairoMakie            # For creating plots
using LaTeXStrings          # For adding LaTeX labels to plots
using Measures              # Use units to specify some dimensions in plots
using Optim                 # For optimisation 
using OrderedCollections    # For OrderedDict so that dictionaries are sorted by insertion order 
using Random                # For setting seeds 
using GaussianProcesses     # For fitting Gaussian processes 
using DifferentialEquations # For differential equations
using StatsBase             # For std
using LinearAlgebra         # For setting number of threads to prevent StackOverflowError
using Setfield              # For modifying immutable structs
using Printf                # For the @sprintf command
using KernelDensity         # For kernel density estimation

#####################################################################
## Set some global parameters for plotting
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
## Define some global parameters for bootstrapping
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
## Study I: Fisher-Kolmogorov Model, 10,000 cells per well
#####################################################################
Random.seed!(51021)
# Select the dataset 
dat = assay_data[1]

# Extract initial conditions
x₀ = dat.Position[dat.Time.==0.0]
u₀ = dat.AvgDens[dat.Time.==0.0]

# Define the functions and parameters
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[1]
D′ = (u, β, p) -> 0.0
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
α = Vector{Float64}([])
β = [310.0] * t_scale / x_scale^2
γ = [0.044] * t_scale
T_params = Vector{Float64}([])
D_params = [1.0]
R_params = [K, 1.0]

# Generate the data 
x, t, u, datgp = EquationLearning.generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
x_pde = copy(x)
t_pde = copy(t)
u_pde = copy(u)

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-1, 2std(u)])
σₙ = log.([1e-5, 2std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# Just some vectors for setting the number of parameters in each function 
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]

# Now do the bootstrapping. We start by seeing if we can learn the scales of the parameters so that we can re-scale for faster optimisation
lowers = [0.001, 0.6]
uppers = [0.5, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)

# Now let us inspect the actual densities
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, diffusion_scales = x_scale^2 / t_scale, reaction_scales = 1 / t_scale)
densityFigures = Figure(fontsize = fontsize, resolution = (800, 400))
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:d
    diffusionDensityAxes[i] = Axis(densityFigures[1, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(densityFigures[1, i+1], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i+1], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
save("figures/simulation_study_initial_fisher_kolmogorov_results.pdf", densityFigures, px_per_unit = 2)

# Now rescale and do more iterations
lowers = [0.7, 0.7]
uppers = [1.3, 1.3]
bootstrap_setup = @set bootstrap_setup.B = 200
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 3
D_params = [323.0] * t_scale / x_scale^2
R_params = [K, 0.044 * t_scale]
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Now plot the results 
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp; level = 0.05, diffusion_scales = D_params[1] * x_scale^2 / t_scale, reaction_scales = R_params[2] / t_scale)
resultFigures = Figure(fontsize = fontsize, resolution = (1200, 800))
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:d
    diffusionDensityAxes[i] = Axis(resultFigures[1, i], xlabel = L"$\beta_%$i$ (μm²/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i], diffusionCIs[i, 1], diffusionCIs[i, 2]),
        titlealign = :left)
    densdat = KernelDensity.kde(dr[i, :])
    lines!(diffusionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
    CI_range = diffusionCIs[i, 1] .< densdat.x .< diffusionCIs[i, 2]
    vlines!(diffusionDensityAxes[i], β * x_scale^2 / t_scale, color = :red, linestyle = :dash)
    band!(diffusionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end
for i = 1:r
    reactionDensityAxes[i] = Axis(resultFigures[2, i], xlabel = L"$\gamma_%$i$ (1/h)", ylabel = "Probability density",
        title = @sprintf("(%s): 95%% CI: (%.3g, %.3g)", alphabet[i+1], reactionCIs[i, 1], reactionCIs[i, 2]),
        titlealign = :left)
    densdat = kde(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x, densdat.density, color = :blue, linewidth = 3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    vlines!(reactionDensityAxes[i], γ / t_scale, color = :red, linestyle = :dash)
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color = (:blue, 0.35))
end

Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp; level = 0.05, x_scale = x_scale, t_scale = t_scale)
diffusionAxis = Axis(resultFigures[1, 2], xlabel = L"$u$ [cells/μm²]", ylabel = L"$D(u)$ [μm²/h]", title = "(c): Diffusion curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
lines!(diffusionAxis, u_vals / x_scale^2, Du_vals[1])
band!(diffusionAxis, u_vals / x_scale^2, Du_vals[3], Du_vals[2], color = (:blue, 0.35))
lines!(diffusionAxis, u_vals / x_scale^2, D.(u_vals, β, [1.0]) .* x_scale^2 / t_scale, color = :red, linestyle = :dash)
reactionAxis = Axis(resultFigures[2, 2], xlabel = L"$u$ [cells/μm²]", ylabel = L"$R(u)$ [1/h]", title = "(d): Reaction curve", linewidth = 1.3, linecolor = :blue, titlealign = :left)
lines!(reactionAxis, u_vals / x_scale^2, Ru_vals[1])
band!(reactionAxis, u_vals / x_scale^2, Ru_vals[3], Ru_vals[2], color = (:blue, 0.35))
lines!(reactionAxis, u_vals / x_scale^2, R.(u_vals, γ, Ref([K, 1.0])) / t_scale, color = :red, linestyle = :dash)

soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp)
err_CI = error_comp(bgp, pde_data, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
dataAxis = Axis(resultFigures[1, 3], xlabel = L"$x$ [μm]", ylabel = L"$u(x, t)$ [cells/μm²]", title = @sprintf("(e): PDE curves with spline ICs\nError: (%.3g, %.3g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(dataAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp)
err_CI = error_comp(bgp, pde_gp, x_pde, t_pde, u_pde)
M = length(bgp.pde_setup.δt)
GPAxis = Axis(resultFigures[2, 3], xlabel = L"$x$ [μm]", ylabel = L"$u(x, t)$ [cells/μm²]", title = @sprintf("(f): PDE curves with sampled ICs\nError: (%.3g, %.3g)", err_CI[1], err_CI[2]), titlealign = :left)
@views for j in 1:M
    lines!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2, color = colors[j])
    band!(GPAxis, bgp.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2, soln_vals_lower[:, j] / x_scale^2, color = (colors[j], 0.35))
    CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp.pde_setup.δt[j]] / x_scale^2, color = colors[j], markersize = 3)
end
Legend(resultFigures[1:2, 4], [values(legendentries)...], [keys(legendentries)...], "Time (h)", orientation = :vertical, labelsize = fontsize, titlesize = fontsize, titleposition = :top)
save("figures/simulation_study_final_fisher_kolmogorov_results.pdf", resultFigures, px_per_unit = 2)

#####################################################################
## Study II: Fisher-Kolmogorov Model, 10,000 cells per well
#####################################################################
Random.seed!(5103821)
# Select the dataset 
dat = assay_data[6]

# Extract initial conditions
x₀ = dat.Position[dat.Time.==0.0]
u₀ = dat.AvgDens[dat.Time.==0.0]

# Define the functions and parameters
T = (t, α, p) -> 1.0/(1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
α = [-4.0651, 0.4166 * t_scale]
β = [2900.0] * t_scale / x_scale^2
γ = [0.0951] * t_scale
T_params = [1.0, 1.0]
D_params = [K, 1.0]
R_params = [K, 1.0]

# Generate the data 
x, t, u, datgp = EquationLearning.generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
x_pde = copy(x)
t_pde = copy(t)
u_pde = copy(u)

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-1, 2std(u)])
σₙ = log.([1e-5, 2std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, GP_Restarts, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts, μ, L, nugget, gp)

# Define functions and parameters to try and learn (misspecified!)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[2] + β[2] * p[3] * (u / p[1])
D′ = (u, β, p) -> β[2] * p[3] / p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
α₀ = Vector{Float64}([])
β₀ = [1.0, 1.0]
γ₀ = [1.0]
T_params = Vector{Float64}([])
D_params = [K, 1.0, 1.0]
R_params = [K, 1.0]

# Now do the bootstrapping. We start by seeing if we can learn the scales of the parameters so that we can re-scale for faster optimisation
lowers = [0.001, 0.001, 0.6]
uppers = [0.5, 0.5, 3.5]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4
bgp = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)

