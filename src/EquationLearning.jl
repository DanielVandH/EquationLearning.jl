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
using LSODA 
using LaTeXStrings 
using LatinHypercubeSampling 
using LinearAlgebra 
using Measures 
using MuladdMacro 
using ODEInterfaceDiffEq 
using Optim 
using Plots 
using PreallocationTools 
using Printf 
using Random 
using StatsBase 
using StatsPlots    
using Sundials
# using RecipesBase

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

end
