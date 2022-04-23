using EquationLearning
using Test
using Random
using StatsBase
using GaussianProcesses
using Distributions
using LinearAlgebra
using LatinHypercubeSampling
using OrderedCollections
using PreallocationTools
using Setfield
using DifferentialEquations
using Printf
using KernelDensity
using FastGaussQuadrature
using StructEquality
using Optim
using CairoMakie
using Sundials
LinearAlgebra.BLAS.set_num_threads(1)
import Base: ==
function ==(x::PreallocationTools.DiffCache, y::PreallocationTools.DiffCache)
    x.du == y.du && x.dual_du == y.dual_du
end

function ==(x::Missing, y::Missing)
    true
end

include("utils.jl")
include("gps.jl")
include("prealloc_helpers.jl")
include("basis_function_evals.jl")
include("struct_constructors.jl")
include("bootstrap_fncs.jl")

#(:delayBases, :diffusionBases, :reactionBases, :gp, 
#:zvals, :Xₛ, :Xₛⁿ, :bootₓ, :bootₜ, :T, :D, :D′, :R, :R′, 
#:D_params, :R_params, :T_params, :μ, :L, :gp_setup, 
#:bootstrap_setup, :pde_setup)

#soln_vals_mean, soln_vals_lower, soln_vals_upper = pde_values(pde_data, bgp2)
