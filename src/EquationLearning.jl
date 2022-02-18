module EquationLearning

using Todo

todo"Use RecipesBase.jl to define plotting methods."
todo"Parallelise most of the computations to significantly improve the runtime."
todo"Add Bayes PDE-Find from Martina-Perez et al. (2021)."
todo"Add the biologically informed neural networks from Lagergren et al. (2020)."
todo"Add some detailed unit tests."
todo"Remove unnecessary dependencies and work on precompilation time."
todo"Add back in the Jacobian computation in the PDEs."
todo"Update documentation of precompute_gp_mean!."
todo"Update documentation of GP_Setup."

#####################################################################
## Required packages
#####################################################################

using DataFrames 
using Dierckx 
using DifferentialEquations 
using FastGaussQuadrature 
using ForwardDiff 
using GaussianProcesses 
using LaTeXStrings 
using LatinHypercubeSampling 
using LinearAlgebra 
using MuladdMacro 
using ODEInterfaceDiffEq 
using Optim 
using CairoMakie
using PreallocationTools 
using Printf 
using StatsBase 
using Random
using StatsPlots 
using KernelDensity   
using Sundials
using Ipopt 
using JuMP 
using MultiJuMP

#####################################################################
## Load files 
#####################################################################

include("structs.jl")           # Definining certain structures
include("gps.jl")               # Fitting Gaussian processes 
include("pdes.jl")              # Working with PDEs 
include("bootstrapping.jl")     # Bootstrapping functions 
include("optimisation.jl")      # Functions for optimising parameters 
include("utils.jl")             # Extra utility functions 
include("plot_results.jl")      # Functions for plotting results from bootstrapping 
include("synthetic_data.jl")    # Function for generating synthetic data 

#####################################################################
## Export functions
#####################################################################

export bootstrap_gp, boot_pde_solve, curve_results, density_results, pde_results, GP_Setup, Bootstrap_Setup, PDE_Setup

end
