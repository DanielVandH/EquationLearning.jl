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
using PrettyTables

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
include("display.jl")            # Displaying results

#####################################################################
## Export functions
#####################################################################

## Structs  
export GP_Setup, Bootstrap_Setup, PDE_Setup, BootResults, BasisBootResults, AllResults

## Bootstrapping  
export bootstrap_gp, basis_bootstrap_gp, update_results

## Comparison 
export AIC, compare_AICs

## GPs  
export fit_GP, compute_joint_GP, precompute_gp_mean

## PDEs   
export boot_pde_solve, error_comp

## Plotting 
export density_values, density_results, curve_values, curve_results, pde_values, pde_results, delay_product

## Synthetic Data  
export generate_data

end
