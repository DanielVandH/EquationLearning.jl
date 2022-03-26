module EquationLearning

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
using Setfield
using KernelDensity   
using Sundials
using StaticArrays
using FLoops 
using SharedArrays

#####################################################################
## Load files 
#####################################################################

include("structs.jl")               # Definining certain structures
include("gps.jl")                   # Fitting Gaussian processes 
include("pdes.jl")                  # Working with PDEs 
include("bootstrapping.jl")         # Bootstrapping functions 
include("optimisation.jl")          # Functions for optimising parameters 
include("utils.jl")                 # Extra utility functions 
include("plot_results.jl")          # Functions for plotting results from bootstrapping 
include("synthetic_data.jl")        # Function for generating synthetic data 
include("basis_bootstrapping.jl")   # Basis function approach to bootstrapping 
include("comparison.jl")            # Model selection

#####################################################################
## Export functions
#####################################################################

export bootstrap_gp, boot_pde_solve, curve_results, 
    density_results, delay_product, pde_results, GP_Setup, 
    Bootstrap_Setup, PDE_Setup, BootResults, density_values, 
    curve_values, pde_values, error_comp, 
    basis_bootstrap_gp, update_results,
    AIC, compare_AICs

end
