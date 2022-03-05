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
## Simulation Study: (1, 1)
#####################################################################

# Extract the parameters 
x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers, gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, α, β, γ = set_parameters(1, assay_data[1], 1, x_scale, t_scale)

# Calibrate 
lowers, uppers = [0.001, 0.001], [2.0, 2.0]
bootstrap_setup = @set bootstrap_setup.B = 10
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 10

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize)

# Perform the full simulation
D_params = 0.006685
R_params = [1.7e-3*x_scale^2, 1.045]
lowers, uppers = [0.95, 0.95], [1.05, 1.05]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 1

#7.37
#8:02
#10:36
@time bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params = D_params, R_params = R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the density values 
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize)
vlines!(diffusionDensityFigure.content[1], β/0.006685, color = :black)
vlines!(reactionDensityFigure.content[1], γ/1.045, color = :black)
delayCurveFigure, diffusionCurveFigure, reactionCurveFigure = curve_results(bgp; fontsize = fontsize)
uvals = LinRange(extrema(u)..., 500)
lines!(diffusionCurveFigure.content[1], uvals, D.(uvals, β, 1.0), color = :black, linewidth = 2)
lines!(reactionCurveFigure.content[1], uvals, R.(uvals, γ, Ref([1.7e-3*x_scale^2, 1.0])), color = :black, linewidth = 2)
pdeDataFigure = pde_results(x_pde, t_pde, u_pde, pde_data, bgp; fontsize = fontsize)
pdeGPFigure = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp; fontsize = fontsize)

#####################################################################
## Simulation Study: (2, 1)
#####################################################################

# Extract the parameters 
x_pde, t_pde, u_pde, x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers, gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, α, β, γ = set_parameters(2, assay_data[1], 1, x_scale, t_scale)

# Calibrate 
lowers, uppers = [0.01, 0.01], [2.0, 2.0]
bootstrap_setup = @set bootstrap_setup.B = 20
bootstrap_setup = @set bootstrap_setup.show_losses = true
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 4

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params, R_params, T_params, verbose = false)
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize)

# Perform the full simulation
D_params = [1.7e-3*x_scale^2, 0.04]
R_params = [1.7e-3*x_scale^2, 1.06]
lowers, uppers = [0.9, 0.9], [1.5, 1.5]
bootstrap_setup = @set bootstrap_setup.B = 100
bootstrap_setup = @set bootstrap_setup.show_losses = false
bootstrap_setup = @set bootstrap_setup.Optim_Restarts = 3

bgp = bootstrap_gp(x, t, u, T, D, D′, R, α₀, β₀, γ₀, lowers, uppers; gp_setup, bootstrap_setup, optim_setup, pde_setup, D_params = D_params, R_params = R_params, T_params, verbose = false)
pde_data = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "data")
pde_gp = boot_pde_solve(bgp, x_pde, t_pde, u_pde; ICType = "gp")

# Plot the density values 
delayDensityFigure, diffusionDensityFigure, reactionDensityFigure = density_results(bgp; fontsize = fontsize)
vlines!(diffusionDensityFigure.content[1], β/0.04, color = :black)
vlines!(reactionDensityFigure.content[1], γ/1.06, color = :black)
delayCurveFigure, diffusionCurveFigure, reactionCurveFigure = curve_results(bgp; fontsize = fontsize)
uvals = LinRange(extrema(u)..., 500)
lines!(diffusionCurveFigure.content[1], uvals, D.(uvals, β, Ref([1.7e-3*x_scale^2, 1.0])), color = :black, linewidth = 2)
lines!(reactionCurveFigure.content[1], uvals, R.(uvals, γ, Ref([1.7e-3*x_scale^2, 1.0])), color = :black, linewidth = 2)
pdeDataFigure = pde_results(x_pde, t_pde, u_pde, pde_data, bgp; fontsize = fontsize)
pdeGPFigure = pde_results(x_pde, t_pde, u_pde, pde_gp, bgp; fontsize = fontsize)

