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
using Sundials              # For CVODE_BDF

#####################################################################
## Set some global parameters for plotting
#####################################################################

fontsize = 20
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
unscaled_K = 1.7e-3 
K = unscaled_K * x_scale^2 # Carrying capacity density as estimated from Jin et al. (2016).

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
N_thin = 38
meshPoints = LinRange(25.0 / x_scale, 1875.0 / x_scale, 50)
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
init_weight = 10.0
bootstrap_setup = EquationLearning.Bootstrap_Setup(bootₓ, bootₜ, B, τ, Optim_Restarts, constrained, obj_scale_GLS, obj_scale_PDE, init_weight, show_losses)

## Setup the GP parameters 
num_restarts = 250
ℓₓ = log.([1e-7, 2.0])
ℓₜ = log.([1e-7, 2.0])
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
## Study II: Fisher-Kolmogorov Model with delay, 10,000 cells per well
#####################################################################

bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1

Random.seed!(5106221)
# Select the dataset 
dat = assay_data[1]

# Extract initial conditions
x₀ = dat.Position[dat.Time.==0.0]
u₀ = dat.AvgDens[dat.Time.==0.0]

# Define the functions and parameters
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
D′ = (u, β, p) -> 0.0
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
α = [-1.50, 0.31 * t_scale]
β = [571.0] * t_scale / x_scale^2
γ = [0.081] * t_scale
T_params = [1.0, 1.0]
D_params = [1.0]
R_params = [K, 1.0]

# Generate the data 
x, t, u, datgp = EquationLearning.generate_data(x₀, u₀, T, D, R, D′, R′, α, β, γ, δt, finalTime; N, LHS, RHS, alg, N_thin, num_restarts, D_params, R_params, T_params)
x_pde = copy(x)
t_pde = copy(t)
u_pde = copy(u)

# Compute the mean vector and Cholesky factor for the GP 
σ = log.([1e-6, 7std(u)])
σₙ = log.([1e-6, 7std(u)])
gp, μ, L = EquationLearning.precompute_gp_mean(x, t, u, ℓₓ, ℓₜ, σ, σₙ, nugget, 250, bootstrap_setup)
gp_setup = EquationLearning.GP_Setup(u; ℓₓ, ℓₜ, σ, σₙ, GP_Restarts=250, μ, L, nugget, gp)

# Plot the actual GP
Σ = L * transpose(L)
fig = Figure(fontsize=fontsize, resolution=(800, 400))
ax = Axis(fig[1, 1], xlabel=L"$x$ (μm)", ylabel=L"u(x, t)/K")
lower = μ .- 2sqrt.(diag(Σ))
upper = μ .+ 2sqrt.(diag(Σ))
for (s, T) in enumerate(unique(t))
    scatter!(ax, x[t.==T] * x_scale, u[t.==T] / x_scale^2 / unscaled_K, color=colors[s], markersize=3)
    idx = findmin(abs.(bootₜ .- T))[2]
    range = ((idx-1)*nₓ+1):(idx*nₓ)
    lines!(ax, bootₓ * x_scale, μ[range] / x_scale^2 / unscaled_K, color=colors[s])
    band!(ax, bootₓ * x_scale, upper[range] / x_scale^2 / unscaled_K, lower[range] / x_scale^2 / unscaled_K, color=(colors[s], 0.35))
end
CairoMakie.ylims!(ax, 0.0, 1.3)
Legend(fig[1, 2], [values(legendentries)...], [keys(legendentries)...], L"$t$ (h)", orientation=:vertical, labelsize=fontsize, titlesize=fontsize, titleposition=:top)
save("figures/simulation_study_delay_fisher_kolmogorov_model_gp_data.pdf", fig, px_per_unit=2)

# Model 1: Correctly specified 
Random.seed!(510226345431)
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
D′ = (u, β, p) -> 0.0
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.5, 0.4 * t_scale]
D_params = [525.0 * t_scale / x_scale^2]
R_params = [K, 0.08 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
#lowers = [-2.0, 0.25 * t_scale, 400.0 * t_scale / x_scale^2, 0.02 * t_scale]
#uppers = [-1.0, 0.8 * t_scale, 1000.0 * t_scale / x_scale^2, 0.06 * t_scale]
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bgp1 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false)
pde_gp = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="gp")

delay1, diffusion1, reaction1 = density_results(bgp1; delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=D_params[1] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
delaycurve1, diffusioncurve1, reactioncurve1 = curve_results(bgp1; x_scale, t_scale)
pdeplot1 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp1; x_scale, t_scale)

# Model 2: Incorrectly specified; missing delay 
Random.seed!(202002991)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[1]
D′ = (u, β, p) -> 0.0
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = Vector{Float64}([])
D_params = [500.0 * t_scale / x_scale^2]
R_params = [K, 0.065 * t_scale]
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]
#lowers = [400.0 * t_scale / x_scale^2, 0.02 * t_scale]
#uppers = [1000.0 * t_scale / x_scale^2, 0.06 * t_scale]
lowers = [0.99, 0.99]
uppers = [1.01, 1.01]
bgp2 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false, zvals=bgp1.zvals)
pde_gp = boot_pde_solve(bgp2, x_pde, t_pde, u_pde; ICType="gp")

delay2, diffusion2, reaction2 = density_results(bgp2; delay_scales=nothing, diffusion_scales=D_params[1] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
delaycurve2, diffusioncurve2, reactioncurve2 = curve_results(bgp2; x_scale, t_scale)
pdeplot2 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp2; x_scale, t_scale)

# Model 3: Incorrectly specified; incorrect diffusion mechanism 
Random.seed!(20636590991)
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.5, 0.25 * t_scale]
D_params = [K, 550.0 * t_scale / x_scale^2]
R_params = [K, 0.08 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0]
γ₀ = [1.0]
#lowers = [-2.0, 0.25 * t_scale, 800.0 * t_scale / x_scale^2, 0.02 * t_scale]
#uppers = [-1.0, 0.8 * t_scale, 2000.0 * t_scale / x_scale^2, 0.07 * t_scale]
lowers = [0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01]
bgp3 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false, zvals=bgp1.zvals)
pde_gp = boot_pde_solve(bgp3, x_pde, t_pde, u_pde; ICType="gp")

delay3, diffusion3, reaction3 = density_results(bgp3; delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=D_params[2] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
delaycurve3, diffusioncurve3, reactioncurve3 = curve_results(bgp3; x_scale, t_scale)
pdeplot3 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp3; x_scale, t_scale)

# Model 4: Incorrectly specified; incorrect diffusion mechanism and no delay
Random.seed!(2063691)
T = (t, α, p) -> 1.0
D = (u, β, p) -> β[1] * p[2] * (u / p[1])
D′ = (u, β, p) -> β[1] * p[2] / p[1]
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = Vector{Float64}([])
D_params = [K, 2200.0 * t_scale / x_scale^2]
R_params = [K, 0.07 * t_scale]
α₀ = Vector{Float64}([])
β₀ = [1.0]
γ₀ = [1.0]
#lowers = [800.0 * t_scale / x_scale^2, 0.02 * t_scale]
#uppers = [2000.0 * t_scale / x_scale^2, 0.07 * t_scale]
lowers = [0.99, 0.99]
uppers = [1.01, 1.01]
bgp4 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false, zvals=bgp1.zvals)
pde_gp = boot_pde_solve(bgp4, x_pde, t_pde, u_pde; ICType="gp")

delay4, diffusion4, reaction4 = density_results(bgp4; delay_scales=nothing, diffusion_scales=D_params[2] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
delaycurve4, diffusioncurve4, reactioncurve4 = curve_results(bgp4; x_scale, t_scale)
pdeplot4 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp4; x_scale, t_scale)

# Model 5: Generalised Porous-FKPP
Random.seed!(2065333691)
T = (t, α, p) -> 1 / (1 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> u > 0 ? β[1] * p[2] + β[2] * p[3] * (u / p[1])^(β[3] * p[4]) : β[1] * p[2]
D′ = (u, β, p) -> u > 0 ? β[2] * p[3] * β[3] * p[4] * (u / p[1])^(β[3] * p[4] - 1) / p[1] : 0.0
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.5, 0.2 * t_scale]
D_params = [K, 250.0 * t_scale / x_scale^2, 350.0 * t_scale / x_scale^2, 0.55]
R_params = [K, 0.08 * t_scale]
α₀ = [1.0, 1.0]
β₀ = [1.0, 1.0, 1.0]
γ₀ = [1.0]
#lowers = [-3.0, 0.15 * t_scale, 200.0 * t_scale / x_scale^2, 0.02 * t_scale]
#uppers = [0.0, 1.1 * t_scale, 1000.0 * t_scale / x_scale^2, 0.08 * t_scale]
lowers = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
uppers = [1.01, 1.01, 1.01, 1.01, 1.01, 1.01]
bgp5 = bootstrap_gp(x, t, u, T, D, D′, R, R′, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose=false, zvals=bgp1.zvals)
pde_gp = boot_pde_solve(bgp5, x_pde, t_pde, u_pde; ICType="gp")

delay5, diffusion5, reaction5 = density_results(bgp5; delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=[D_params[2] * x_scale^2 / t_scale, D_params[3] * x_scale^2 / t_scale, D_params[4]], reaction_scales=R_params[2] / t_scale)
delaycurve5, diffusioncurve5, reactioncurve5 = curve_results(bgp5; x_scale, t_scale)
pdeplot5 = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp5; x_scale, t_scale)

# Model comparisons 
Random.seed!(12929201)
T_params1 = [-1.5, 0.4 * t_scale]
D_params1 = [525.0 * t_scale / x_scale^2]
R_params1 = [K, 0.08 * t_scale]
T_params2 = Vector{Float64}([])
D_params2 = [500.0 * t_scale / x_scale^2]
R_params2 = [K, 0.065 * t_scale]
T_params3 = [-1.5, 0.25 * t_scale]
D_params3 = [K, 550.0 * t_scale / x_scale^2]
R_params3 = [K, 0.08 * t_scale]
T_params4 = Vector{Float64}([])
D_params4 = [K, 2200.0 * t_scale / x_scale^2]
R_params4 = [K, 0.07 * t_scale]
T_params5 = [-1.5, 0.2 * t_scale]
D_params5 = [K, 250.0 * t_scale / x_scale^2, 350.0 * t_scale / x_scale^2, 0.55]
R_params5 = [K, 0.08 * t_scale]
delay_scales1, diffusion_scales1, reaction_scales1 = [T_params1[1], T_params1[2] / t_scale], D_params1[1] * x_scale^2 / t_scale, R_params1[2] / t_scale
delay_scales2, diffusion_scales2, reaction_scales2 = nothing, D_params2[1] * x_scale^2 / t_scale, R_params2[2] / t_scale
delay_scales3, diffusion_scales3, reaction_scales3 = [T_params3[1], T_params3[2] / t_scale], D_params3[2] * x_scale^2 / t_scale, R_params3[2] / t_scale
delay_scales4, diffusion_scales4, reaction_scales4 = nothing, D_params4[2] * x_scale^2 / t_scale, R_params4[2] / t_scale
delay_scales5, diffusion_scales5, reaction_scales5 = [T_params5[1], T_params5[2] / t_scale], [D_params5[2] * x_scale^2 / t_scale, D_params5[3] * x_scale^2 / t_scale, D_params5[4]], R_params5[2] / t_scale
delay_scales = [delay_scales1, delay_scales2, delay_scales3, delay_scales4, delay_scales5]
diffusion_scales = [diffusion_scales1, diffusion_scales2, diffusion_scales3, diffusion_scales4, diffusion_scales5]
reaction_scales = [reaction_scales1, reaction_scales2, reaction_scales3, reaction_scales4, reaction_scales5]
res = AllResults(x_pde, t_pde, u_pde, bgp1, bgp2, bgp3, bgp4, bgp5; delay_scales, diffusion_scales, reaction_scales, x_scale, t_scale) # paper uses different column order
#specifically, the order is 1 -> 2, 2 -> 1, 3 -> 4, 4 -> 3, 5 -> 5 (script -> paper)

# Plot the bgp1 results 
pde_gp = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="gp")
pde_data = boot_pde_solve(bgp1, x_pde, t_pde, u_pde; ICType="data")
T = (t, α, p) -> 1.0 / (1.0 + exp(-α[1] * p[1] - α[2] * p[2] * t))
D = (u, β, p) -> β[1] * p[1]
D′ = (u, β, p) -> 0.0
R = (u, γ, p) -> γ[1] * p[2] * u * (1.0 - u / p[1])
R′ = (u, γ, p) -> γ[1] * p[2] - 2.0 * γ[1] * p[2] * u / p[1]
T_params = [-1.5, 0.4 * t_scale]
D_params = [525.0 * t_scale / x_scale^2]
R_params = [K, 0.08 * t_scale]
unscaled_α = [α[1], α[2] / t_scale]
unscaled_β = β[1] * x_scale^2 / t_scale
unscaled_γ = γ[1] / t_scale
trv, dr, rr, tt, d, r, delayCIs, diffusionCIs, reactionCIs = density_values(bgp1; level=0.05, delay_scales=[T_params[1], T_params[2] / t_scale], diffusion_scales=D_params[1] * x_scale^2 / t_scale, reaction_scales=R_params[2] / t_scale)
resultFigures = Figure(fontsize=fontsize, resolution=(1800, 800))
delayDensityAxes = Vector{Axis}(undef, tt)
diffusionDensityAxes = Vector{Axis}(undef, d)
reactionDensityAxes = Vector{Axis}(undef, r)
for i = 1:tt
    delayDensityAxes[i] = Axis(resultFigures[1, i], xlabel=i == 1 ? L"$\alpha_%$i$" : L"$\alpha_%$i$ (1/h)", ylabel="Probability density",
        title=@sprintf("(%s): 95%% CI: (%.4g, %.4g)", alphabet[i], delayCIs[i, 1], delayCIs[i, 2]),
        titlealign=:left)
    densdat = KernelDensity.kde(trv[i, :])
    vlines!(delayDensityAxes[i], unscaled_α[i], color=:red, linestyle=:dash)
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
    vlines!(diffusionDensityAxes[i], unscaled_β[i], color=:red, linestyle=:dash)
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
    vlines!(reactionDensityAxes[i], unscaled_γ[i], color=:red, linestyle=:dash)
    in_range = minimum(rr[i, :]) .< densdat.x .< maximum(rr[i, :])
    lines!(reactionDensityAxes[i], densdat.x[in_range], densdat.density[in_range], color=:blue, linewidth=3)
    CI_range = reactionCIs[i, 1] .< densdat.x .< reactionCIs[i, 2]
    band!(reactionDensityAxes[i], densdat.x[CI_range], densdat.density[CI_range], zeros(count(CI_range)), color=(:blue, 0.35))
end

Tu_vals, Du_vals, Ru_vals, u_vals, t_vals = curve_values(bgp1; level=0.05, x_scale=x_scale, t_scale=t_scale)
delayAxis = Axis(resultFigures[1, 3], xlabel=L"$t$ (h)", ylabel=L"$T(t)$", title="(c): Delay curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
lines!(delayAxis, t_vals * t_scale, Tu_vals[1])
band!(delayAxis, t_vals * t_scale, Tu_vals[3], Tu_vals[2], color=(:blue, 0.35))
lines!(delayAxis, t_vals * t_scale, T.(t_vals, Ref(α), Ref([1.0, 1.0])), color=:red, linestyle=:dash)
diffusionAxis = Axis(resultFigures[2, 3], xlabel=L"$u$ (cells/μm²)", ylabel=L"$T(t)D(u)$ (μm²/h)", title="(f): Nonlinear diffusivity curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
Du_vals0 = delay_product(bgp1, 0.0; type="diffusion", x_scale=x_scale, t_scale=t_scale)
Du_vals12 = delay_product(bgp1, 12.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
Du_vals24 = delay_product(bgp1, 24.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
Du_vals36 = delay_product(bgp1, 36.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
Du_vals48 = delay_product(bgp1, 48.0 / t_scale; type="diffusion", x_scale=x_scale, t_scale=t_scale)
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
lines!(diffusionAxis, u_vals / x_scale^2, D.(u_vals, β, Ref([1.0])) .* x_scale^2 / t_scale, color=:red, linestyle=:dash)
reactionAxis = Axis(resultFigures[3, 3], xlabel=L"$u$ (cells/μm²)", ylabel=L"$T(t)R(u)$ (cells/μm²h)", title="(i): Reaction curve", linewidth=1.3, linecolor=:blue, titlealign=:left)
Ru_vals0 = delay_product(bgp1, 0.0; type="reaction", x_scale=x_scale, t_scale=t_scale)
Ru_vals12 = delay_product(bgp1, 12.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
Ru_vals24 = delay_product(bgp1, 24.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
Ru_vals36 = delay_product(bgp1, 36.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
Ru_vals48 = delay_product(bgp1, 48.0 / t_scale; type="reaction", x_scale=x_scale, t_scale=t_scale)
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
lines!(reactionAxis, u_vals / x_scale^2, R.(u_vals, γ, Ref([K, 1.0])) / t_scale, color=:red, linestyle=:dash)

soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp1)
err_CI = error_comp(bgp1, pde_data, x_pde, t_pde, u_pde)
M = length(bgp1.pde_setup.δt)
dataAxis = Axis(resultFigures[3, 1], xlabel=L"$x$ (μm)", ylabel=L"u(x, t)/K", title=@sprintf("(g): PDE (Spline IC). Error: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign=:left)
@views for j in 1:M
    lines!(dataAxis, bgp1.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2 / unscaled_K, color=colors[j])
    band!(dataAxis, bgp1.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2 / unscaled_K, soln_vals_lower[:, j] / x_scale^2 / unscaled_K, color=(colors[j], 0.35))
    CairoMakie.scatter!(dataAxis, x_pde[t_pde.==bgp1.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp1.pde_setup.δt[j]] / x_scale^2 / unscaled_K, color=colors[j], markersize=3)
end
soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_gp, bgp1)
err_CI = error_comp(bgp1, pde_gp, x_pde, t_pde, u_pde)
M = length(bgp1.pde_setup.δt)
GPAxis = Axis(resultFigures[3, 2], xlabel=L"$x$ (μm)", ylabel=L"u(x, t)/K", title=@sprintf("(h): PDE (Sampled IC). Error: (%.4g, %.4g)", err_CI[1], err_CI[2]), titlealign=:left)
@views for j in 1:M
    lines!(GPAxis, bgp1.pde_setup.meshPoints * x_scale, soln_vals_mean[:, j] / x_scale^2 / unscaled_K, color=colors[j])
    band!(GPAxis, bgp1.pde_setup.meshPoints * x_scale, soln_vals_upper[:, j] / x_scale^2 / unscaled_K, soln_vals_lower[:, j] / x_scale^2 / unscaled_K, color=(colors[j], 0.35))
    CairoMakie.scatter!(GPAxis, x_pde[t_pde.==bgp1.pde_setup.δt[j]] * x_scale, u_pde[t_pde.==bgp1.pde_setup.δt[j]] / x_scale^2 / unscaled_K, color=colors[j], markersize=3)
end
Legend(resultFigures[1:3, 4], [values(legendentries)...], [keys(legendentries)...], L"$t$ (h)", orientation=:vertical, labelsize=fontsize, titlesize=fontsize, titleposition=:top)
#save("figures/simulation_study_bgp1_final_results_2.pdf", resultFigures, px_per_unit=2)
