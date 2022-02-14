#####################################################################
## Script description: gps.jl 
##
## This script contains certain functions used for fitting Gaussian 
## processes, primarily  leveraging the functions provided by 
## GaussianProcesses.jl.
##
## The following functions are defined:
##  - opt_restart!: Method for restarting the optimiser for the 
##      Gaussian process with many initial guesses.
##  - fit_GP: Fits a Gaussian process to some provided data.
##  - dkxⱼ, d⁴kxᵢ²xⱼ², etc: Computing derivatives of the squared 
##      exponential kernel, for use with compute_joint_GP.
##  - compute_joint_GP: Computes the arrays required for the joint 
##      Gaussian process [f, fₜ, fₓ, fₓₓ]. 
##  - draw_gp!: Draw a random sample from a Gaussian process.
##
#####################################################################

"""
    opt_restart!(gp, ℓₓ, ℓₜ, σ, σₙ; num_restarts = 50)

Given a Gaussian process `gp`, fit many new Gaussian processes with new initial estimates for the hyperparameters. The 
initial estimates are chosen based on provided ranges for the hyperparameters and Latin hypercube sampling. See also [`fit_GP`](@ref).

# Arguments 
- `gp`: A Gaussian process object, fitted uses the `GaussianProcesses.jl` package. 
- `ℓₓ`: A 2-vector giving the lower and upper bounds for the initial estimates of `ℓₓ` (defined on a log scale).
- `ℓₜ`: A 2-vector giving the lower and upper bounds for the initial estimates of `ℓₜ` (defined on a log scale).
- `σ`: A 2-vector giving the lower and upper bounds for the initial estimates of `σ` (defined on a log scale).
- `σₙ`: A 2-vector giving the lower and upper bounds for the initial estimates of `σₙ` (defined on a log scale).

# Keyword Arguments 
- `num_restarts = 50`: The number of restarts to perform.
"""
function opt_restart!(gp, ℓₓ, ℓₜ, σ, σₙ; num_restarts = 50)
    @assert length(ℓₓ) == length(ℓₜ) == length(σ) == length(σₙ) == 2 "The provided hyperparameters must be given as vectors of length 2 providing upper and lower bounds."
    @assert issorted(ℓₓ) && issorted(ℓₜ) && issorted(σ) && issorted(σₙ) "The provided ranges for the hyperparameters must be given as (lower, upper)."

    ## Form the design 
    plan, _ = LHCoptim(num_restarts, length(GaussianProcesses.get_params(gp)), 1000)

    # Scale the design into the provided intervals 
    new_params = scaleLHC(plan, [(ℓₓ[1], ℓₓ[2]), (ℓₜ[1], ℓₜ[2]), (σ[1], σ[2]), (σₙ[1], σₙ[2])])' # Transpose to get a more efficient order for accessing in a loop

    # Optimise
    obj_values = zeros(num_restarts)
    for j = 1:num_restarts
        try
            @views GaussianProcesses.set_params!(gp, new_params[:, j])
            GaussianProcesses.optimize!(gp)
            obj_values[j] = gp.target
        catch err
            println(err)
            obj_values[j] = -Inf
        end
    end
    # Return optimal model 
    opt_model = findmax(obj_values)[2]
    @views GaussianProcesses.set_params!(gp, new_params[:, opt_model])
    GaussianProcesses.optimize!(gp)
    return nothing
end

"""
    fit_GP(x, t, u; <keyword arguments>)

Fits a Gaussian process with data `(x, t)` using the targets in `u`. 

# Arguments 
- `x`: The spatial data. 
- `t`: The temporal data.
- `u`: The targets corresponding to `(x, t)`.

# Keyword Arguments 
- `ℓₓ = log.([1e-4, 1.0])`: A 2-vector giving the lower and upper bounds for the initial estimates of `ℓₓ` (defined on a log scale).
- `ℓₜ = log.([1e-4, 1.0])`: A 2-vector giving the lower and upper bounds for the initial estimates of `ℓₜ` (defined on a log scale).
- `σ = log.([1e-1, 2std(u)])`: A 2-vector giving the lower and upper bounds for the initial estimates of `σ` (defined on a log scale).
- `σₙ = log.([1e-5, 2std(u)])`: A 2-vector giving the lower and upper bounds for the initial estimates of `σₙ` (defined on a log scale).
- `num_restarts = 50`: Number of times to restart the optimiser. See [`Optimise_Restart!`](@ref).

# Outputs 
- `gp`: The fitted Gaussian process.
"""
function fit_GP(x, t, u; ℓₓ = log.([1e-4, 1.0]), ℓₜ = log.([1e-4, 1.0]),
    σ = log.([1e-1, 2std(u)]), σₙ = log.([1e-5, 2std(u)]), num_restarts = 50)
    @assert length(x) == length(t) == length(u) "The provided data (x, t, u) must all be the same length."

    # Define the GP 
    meanFunc = MeanZero()
    covFunc = SE([0.0, 0.0], 0.0) # These numbers don't matter, they get replaced later by opt_restart

    # Ensure that the data are row vectors and that the target is a column vector
    x = Matrix(vec(x)')
    t = Matrix(vec(t)')
    u = vec(u)

    # Now normalise the data 
    xx = scale_unit(x)
    tt = scale_unit(t)
    X = [xx; tt]

    # Now fit the GP
    gp = GPE(X, u, meanFunc, covFunc, -2.0)

    ## Obtain num_restarts random estimates for the hyperparameters
    opt_restart!(gp, ℓₓ, ℓₜ, σ, σₙ; num_restarts = num_restarts)
    return gp
end

@doc "See [`compute_joint_GP`](@ref) for details." @inline function dkxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (x₁ - x₂) / ℓ₁^2
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d²kxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / ℓ₁^4
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline function dktⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (t₁ - t₂) / ℓ₂^2
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline function dktᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(t₁ - t₂) / ℓ₂^2
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d²ktᵢtⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (ℓ₂ * ℓ₂ - t₁ * t₁ + 2t₁ * t₂ - t₂ * t₂) / ℓ₂^4
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline function d²ktᵢxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(t₁ - t₂) * (x₁ - x₂) / (ℓ₁^2 * ℓ₂^2)
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d³tᵢxⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (t₁ - t₂) * (ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / (ℓ₁^4 * ℓ₂^2)
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline function dkxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(x₁ - x₂) / (ℓ₁^2)
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline function d²kxᵢtⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(t₁ - t₂) * (x₁ - x₂) / (ℓ₁^2 * ℓ₂^2)
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d²kxᵢxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / ℓ₁^4
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d³kxᵢxⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (x₁ - x₂) * (3ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / ℓ₁^6
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d²kxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / ℓ₁^4
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d³kxᵢ²tⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(t₁ - t₂) * (ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / (ℓ₁^4 * ℓ₂^2)
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d³kxᵢ²xⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    -(x₁ - x₂) * (3ℓ₁ * ℓ₁ - x₁ * x₁ + 2x₁ * x₂ - x₂ * x₂) / ℓ₁^6
end
@doc "See [`compute_joint_GP`](@ref) for details." @inline @muladd function d⁴kxᵢ²xⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
    (3ℓ₁ * ℓ₁ * ℓ₁ * ℓ₁ - 6ℓ₁ * ℓ₁ * x₁ * x₁ + 12ℓ₁ * ℓ₁ * x₁ * x₂ - 6ℓ₁ * ℓ₁ * x₂ * x₂ + x₁ * x₁ * x₁ * x₁ - 4x₁ * x₁ * x₁ * x₂ + 6x₁ * x₁ * x₂ * x₂ - 4x₁ * x₂ * x₂ * x₂ + x₂ * x₂ * x₂ * x₂) / ℓ₁^8
end

"""
    compute_joint_GP(gp, X̃; nugget = 1e-10)

Computes the mean vector `μ` and Cholesky factor `L` such that `LLᵀ = Σ`, where 

`` \\left[\\begin{array}{c} \\mathbf{f} \\\\ \\frac{\\partial\\mathbf f}{\\partial t} \\\\ \\frac{\\partial\\mathbf f}{\\partial x} \\\\\\frac{\\partial^2\\mathbf f}{\\partial x^2}\\end{array}\\right] \\sim \\mathcal N\\left(\\mathbf{\\mu}, \\mathbf{\\Sigma}\\right).``

# Arguments 
- `gp`: The fitted Gaussian process. 
- `X̃`: Test data for the Gaussian process. 

# Keyword Arguments 
- `nugget = 1e-10`: The term to add to the diagonals of the covariance matrix in case the matrix is not positive definite.

# Outputs 
- `μ`: The mean vector. 
- `L`: The Cholesky factor of the covariance matrix.

# Extended help
The covariance matrices are built without any attention to symmetry. The loops could be optimised by 
e.g. considering only the upper triangular components. 

The covariance matrices are built using separate functions for the derivatives of the kernel function (currently 
only implemented for the squared exponential kernel). These functions are:

- [`dkxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d²kxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`dktⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`dktᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d²ktᵢtⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d²ktᵢxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d³tᵢxⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`dkxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d²kxᵢtⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d²kxᵢxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d³kxᵢxⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d²kxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d³kxᵢ²tⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d³kxᵢ²xⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).
- [`d⁴kxᵢ²xⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)`](@ref).

In these functions, the derivatives are evaluated at `([x₁; t₁], [x₂; t₂])` with length scales 
`(ℓ₁, ℓ₂)` for space and time, respectively. The functions are missing a multiplication by the 
kernel value `cov(gp.kernel, [x₁; t₁], [x₂; t₂])` to allow for it to be done in place more easily. 
Note also that in some of these functions, powers are written as products, e.g. `x³ = x*x*x`, to allow 
for the `@muladd` macro to work (see https://github.com/SciML/MuladdMacro.jl). Since the functions are 
quite small, we use `@inline` on each function to encourage the compiler to inline the function 
in the LLVM.

You may get a `StackOverflowError` from this function. This may be related to https://github.com/JuliaLang/julia/issues/43242,
in which case you can set the number of BLAS threads to 1:
```julia-repl 
julia> ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ()) # In case you need to remember how many threads you used prior
julia> LinearAlgebra.BLAS.set_num_threads(1)
```
"""
function compute_joint_GP(gp, X̃; nugget = 1e-10)
    # Extract the original data matrix 
    X = gp.x
    nₓnₜ = size(X̃, 2)
    NM = size(X, 2)
    ℓ₁, ℓ₂ = gp.kernel.iℓ2 .^ (-1 / 2)

    # The primary covariance matrices 
    Kyy = gp.cK
    Kyf = cov(gp.kernel, X, X̃)
    Kfy = cov(gp.kernel, X̃, X)
    Kff = cov(gp.kernel, X̃, X̃)

    # The remaining covariance matrices between data and test data
    Ky_∂f∂t = copy(Kyf)
    Ky_∂f∂x = copy(Kyf)
    Ky_∂²f∂x² = copy(Kyf)
    K∂f∂t_y = copy(Kfy)
    K∂f∂x_y = copy(Kfy)
    K∂²f∂x²_y = copy(Kfy)
    for j = 1:nₓnₜ
        for i = 1:NM
            x₁, t₁, x₂, t₂ = X[1, i], X[2, i], X̃[1, j], X̃[2, j]
            Ky_∂f∂t[i, j] *= dktⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            Ky_∂f∂x[i, j] *= dkxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            Ky_∂²f∂x²[i, j] *= d²kxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
        end
    end
    for j = 1:NM
        for i = 1:nₓnₜ
            x₁, t₁, x₂, t₂ = X̃[1, i], X̃[2, i], X[1, j], X[2, j]
            K∂f∂t_y[i, j] *= dktᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂x_y[i, j] *= dkxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂²f∂x²_y[i, j] *= d²kxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
        end
    end

    # Covariance matrices between test data 
    Kf_∂f∂t = copy(Kff)
    Kf_∂f∂x = copy(Kff)
    Kf_∂²f∂x² = copy(Kff)
    K∂f∂t_f = copy(Kff)
    K∂f∂t_∂f∂t = copy(Kff)
    K∂f∂t_∂f∂x = copy(Kff)
    K∂f∂t_∂²f∂x² = copy(Kff)
    K∂f∂x_f = copy(Kff)
    K∂f∂x_∂f∂t = copy(Kff)
    K∂f∂x_∂f∂x = copy(Kff)
    K∂f∂x_∂²f∂x² = copy(Kff)
    K∂²f∂x²_f = copy(Kff)
    K∂²f∂x²_∂f∂t = copy(Kff)
    K∂²f∂x²_∂f∂x = copy(Kff)
    K∂²f∂x²_∂²f∂x² = copy(Kff)
    for j = 1:nₓnₜ
        for i = 1:nₓnₜ
            x₁, t₁, x₂, t₂ = X̃[1, i], X̃[2, i], X̃[1, j], X̃[2, j]
            Kf_∂f∂t[i, j] *= dktⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            Kf_∂f∂x[i, j] *= dkxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            Kf_∂²f∂x²[i, j] *= d²kxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂t_f[i, j] *= dktᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂t_∂f∂t[i, j] *= d²ktᵢtⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂t_∂f∂x[i, j] *= d²ktᵢxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂t_∂²f∂x²[i, j] *= d³tᵢxⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂x_f[i, j] *= dkxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂x_∂f∂t[i, j] *= d²kxᵢtⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂x_∂f∂x[i, j] *= d²kxᵢxⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂f∂x_∂²f∂x²[i, j] *= d³kxᵢxⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂²f∂x²_f[i, j] *= d²kxᵢ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂²f∂x²_∂f∂t[i, j] *= d³kxᵢ²tⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂²f∂x²_∂f∂x[i, j] *= d³kxᵢ²xⱼ(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
            K∂²f∂x²_∂²f∂x²[i, j] *= d⁴kxᵢ²xⱼ²(x₁, t₁, x₂, t₂, ℓ₁, ℓ₂)
        end
    end

    # Now combine the results 
    Σ₁₁ = Symmetric(Kyy.mat)
    Σ₁₂ = hcat(Kyf, Ky_∂f∂t, Ky_∂f∂x, Ky_∂²f∂x²)
    Σ₂₁ = vcat(Kfy, K∂f∂t_y, K∂f∂x_y, K∂²f∂x²_y)
    Σ₂₂ = Symmetric([Kff Kf_∂f∂t Kf_∂f∂x Kf_∂²f∂x²
        K∂f∂t_f K∂f∂t_∂f∂t K∂f∂t_∂f∂x K∂f∂t_∂²f∂x²
        K∂f∂x_f K∂f∂x_∂f∂t K∂f∂x_∂f∂x K∂f∂x_∂²f∂x²
        K∂²f∂x²_f K∂²f∂x²_∂f∂t K∂²f∂x²_∂f∂x K∂²f∂x²_∂²f∂x²])

    # Compute the mean vector 
    μ = Σ₂₁ * gp.alpha

    # Compute the covariance matrix and Cholesky factor 
    Σ = Symmetric(Σ₂₂ - Σ₂₁ * (Σ₁₁ \ Σ₁₂))
    local L # So it's kept outside of try
    try
        Σ, chol = GaussianProcesses.make_posdef!(Σ; nugget = nugget)
        L = chol.L
    catch
        Σvar = diag(Σ)
        for i = 1:size(Σ, 2)
            Σ[i, i] += nugget * Σvar[i] # add nugget to correlation matrix instead!
        end
        Σ, chol = GaussianProcesses.make_posdef!(Σ; nugget = nugget)
        L = chol.L
    end

    # Return 
    return μ, L
end

"""
    draw_gp!(F, μ, L, z, ℓz) 
    
Draws a random sample from a Gaussian process with mean `μ` and 
covariance matrix `Σ = LLᵀ` corresponding to `z ∼ N(0, I)`.

# Arguments
- `F`: Cache array used for storing the random sample.
- `μ`: The mean vector.
- `L`: The Cholesky factor of the covariance matrix.
- `z`: The random sample from `z ∼ N(0, I)`.
- `ℓz`: Cache array for storing the result of the matrix-vector product `Lz`.

# Outputs 
The random sample is updated in-place into `F`.
"""
@inline function draw_gp!(F, μ, L, z, ℓz)
    F .= μ + mul!(ℓz, LowerTriangular(L), z)
    return nothing
end
