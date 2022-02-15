module EquationLearning

using Todo

todo"Use RecipesBase.jl to define plotting methods."
todo"Parallelise most of the computations to significantly improve the runtime."
todo"Add Bayes PDE-Find from Martina-Perez et al. (2021)."
todo"Add the biologically informed neural networks from Lagergren et al. (2020)."
todo"Add some detailed unit tests."
todo"Remove unnecessary dependencies and work on precompilation time."

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
using StatsPlots    
using Sundials

#####################################################################
## Load files 
#####################################################################

include("structs.jl")           # Definining certain structures
include("gps.jl")               # Fitting Gaussian processes 
include("pdes.jl")              # Working with PDEs 
include("recipes.jl")           # Utility functions for working with plots
include("bootstrapping.jl")     # Bootstrapping functions 
include("optimisation.jl")      # Functions for optimising parameters 
include("utils.jl")             # Extra utility functions 
include("plot_results.jl")      # Functions for plotting results from bootstrapping 
include("synthetic_data.jl")    # Function for generating synthetic data 

#####################################################################
## Export functions
#####################################################################

export bootstrap_gp, curve_results, density_results, pde_results

end
