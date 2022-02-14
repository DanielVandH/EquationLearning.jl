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

#####################################################################
## Load files 
#####################################################################

include("gps.jl")
include("pdes.jl")
include("recipes.jl")
include("bootstrapping.jl")
include("optimisation.jl")
include("utils.jl")
include("structs.jl")

#####################################################################
## Export functions
#####################################################################

end
