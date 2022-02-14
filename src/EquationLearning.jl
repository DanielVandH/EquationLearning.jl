module EquationLearning

#####################################################################
## Required packages
#####################################################################

using GaussianProcesses
using Optim
using MuladdMacro
using LSODA
using Dierckx
using LaTeXStrings
using DifferentialEquations
using Sundials
using FastGaussQuadrature
using ODEInterfaceDiffEq
using Plots
using PreallocationTools
using Printf
using Random
using LinearAlgebra
using ForwardDiff
using StatsPlots
using Measures
using LatinHypercubeSampling
using StatsBase
using DataFrames

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
