#####################################################################
## Script description: optimisation.jl
##
## This script contains certain functions used for estimating parameters.
##
## The following functions are defined:
##  - loss_function: Computes the loss function. 
##  - learn_equations!: Estimates the parameter values.
##
#####################################################################

"""
    loss_function(αβγ; <keyword arguments>)

Computes the loss function at `αβγ`.

# Arguments 
- `αβγ`: The parameter values for the delay, diffusion, and reaction terms, given in the form `[α, β, γ]`.

# Keyword Arguments
- `u`: The density data used for fitting the Gaussian process. See [`fit_GP`](@ref).
- `f`: Computed values of `f(x, t)`.
- `fₓ`: Computed values of `fₓ(x, t)`.
- `fₓₓ`: Computed values of `fₓₓ(x, t)`.
- `fₜ`: Computed values of `fₜ(x, t)`.
- `N`: Number of mesh points being used for solving the PDEs.
- `V`: The volume of each cell in the discretised PDE.
- `Δx`: The spacing between cells in the discretised PDE.
- `LHS`: Vector of the form `[a₀, b₀, c₀]` which specifies the boundary conditions. 
- `RHS`: Vector of the form `[a₁, b₁, c₁]` which specifies the boundary conditions. 
- `T`: The delay function, given in the form `T(t, α...)`.
- `D`: The diffusion function, given in the form `D(u, β...)`.
- `D′`: The derivative of the diffusion function, given in the form `D′(u, β...)`.
- `R`: The reaction function, given in the form `R(u, γ...)`.
- `R′::Function`: The derivative of the reaction function, given in the form `R′(u, γ, R_params)`.
- `initialCondition`: Vector for the initial condition for the PDE.
- `finalTime`: The final time to give the solution to the PDE at.
- `EQLalg`: The algorithm to use for solving the PDE. Cannot be a Sundials algorithm.
- `δt`: A number or a vector specifying the spacing between returned times for the solutions to the PDEs or specific times, respectively.
- `SSEArray`: Cache array for storing the solutions to the PDE. 
- `Du`: Cache array for storing the values of the diffusion function at the mesh points.
- `Ru`: Cache array for storing the values of the reaction function at the mesh points. 
- `D′u`: Cache array used for storing the values of the derivative of the diffusion function `D` at `D′(u)`.
- `R′u`: Cache array used for storing the values of the derivative of the reaction function `R` at `R′(u)`.
- `TuP`: Cache array for storing the values of the delay function at the unscaled times (for the `"GLS"` loss function).
- `DuP`: Cache array for storing the values of the diffusion function at the estimated density values (for the `"GLS"` loss function).
- `RuP`: Cache array for storing the values of the reaction function at the estimated density values (for the `"GLS"` loss function).
- `D′uP`: Cache array for storing the values of the derivative of the diffusion function at the estimated density values (for the `"GLS"` loss function).
- `RuN`: For storing values of the reaction function at Gauss-Legendre quadrature nodes.
- `inIdx`: The indices for the data to use. See [`data_thresholder`](@ref).
- `unscaled_t̃`: The unscaled time values on the bootstrapping grid.
- `tt`: The number of delay parameters.
- `d`: The number of diffusion parameters.
- `r`: The number of reaction parameters.
- `errs`: Cache array for storing the individual error values.
- `MSE`: Cache array for storing the individual squared errors.
- `obj_scale_GLS`: Scale to divide the `GLS` loss by.
- `obj_scale_PDE`: Scale to divide the `PDE` loss by.
- `glnodes`: Gauss-Legendre quadrature nodes.
- `glweights`: Gauss-Legendre quadrature weights.
- `maxf`: The maximum value of `f`.
- `D_params`: Extra parameters for the diffusion function.
- `R_params`: Extra parameters for the reaction function. 
- `T_params`: Extra parameters for the delay function.
- `iterate_idx`: Vector used for indexing the values in `u` corresponding to different times.
- `closest_idx`: Vector used for indexing the points in the PDE's meshpoints that are closest to the actual spatial data used for fitting the Gaussian process.
- `show_losses`: Whether to print the individual loss functions to the REPL during the optimisation process.
- `σₙ`: The standard deviation of the observation noise of the Gaussian process.
- `PDEkwargs...`: The keyword arguments to use in `DifferentialEquations.solve`.

# Extended help
There are two types of loss functions currently considered, namely `"GLS"` and `"PDE"`. For `"PDE"`, the loss function is 

`` \\frac{1}{n_xn_t}\\sum_{i=1}^{n_x}\\sum_{j=1}^{n_t}\\left\\{\\frac{\\partial u_{ij}}{\\partial t} - T(t_j; \\mathbf{\\alpha})\\left[\\frac{\\mathrm{d}D(u_{ij}; \\mathbf{\\beta})}{\\mathrm du}\\left(\\frac{\\partial u_{ij}}{x}\\right)^2 + D(u_{ij}; \\mathbf{\\beta})\\frac{\\partial^2u_{ij}}{\\partial x^2} + R(u_{ij}; \\mathbf{\\gamma})\\right]\\right\\}.``

For `"GLS"`, the loss function is (see Lagergren et al. (2020): https://doi.org/10.1098/rspa.2019.0800)

`` \\frac{1}{NM}\\sum_{i=1}^N\\sum_{j=1}^M \\left(\\frac{\\hat u_{ij} - u_{ij}}{\\σ_{n}\\right)^2 ``,

If the ODE solver returns an error for the given parameter values, `∞` is added to the loss function as a penalty.
"""
function loss_function(αβγ; u,
    f, fₓ, fₓₓ, fₜ,
    N, V, Δx, LHS, RHS,
    T, D, D′, R, R′,
    initialCondition, finalTime, EQLalg, δt, SSEArray, Du, Ru, D′u, R′u, TuP, DuP, RuP, D′uP, RuN,
    inIdx, unscaled_t̃, tt, d, r, errs, MSE, obj_scale_GLS, obj_scale_PDE, glnodes, glweights, maxf,
    D_params, R_params, T_params, iterate_idx, closest_idx, show_losses, σₙ, PDEkwargs...)
    α = @views αβγ[1:tt]
    β = @views αβγ[(tt+1):(tt+d)]
    γ = @views αβγ[(tt+d+1):(tt+d+r)]
    RuN = get_tmp(RuN, first(αβγ)) # This extracts the cache array with the same type as the elements in αβγ
    total_loss = eltype(αβγ)(0.0) # eltype ensures we get the correct type for the dual numbers
    Reaction = u -> R(maxf / 2 * (u + 1), γ, R_params) # This function is taking the interval [0, fmax] into [-1, 1] to use Gauss-Legendre quadrature; ignoring the fmax/2 Jacobian in the integrand since this is positive anyway
    @inbounds for idx in eachindex(glnodes)
        RuN[idx] = Reaction(glnodes[idx])
    end
    Ival = dot(glweights, RuN) # Integrate the reaction curve over [0, fmax] using Gauss-Legendre quadrature
    local GLSC = Inf # So it can be displayed later on if needed - local allows it to enter loops
    local PDEC = Inf
    if Ival < 0.0 # Negative area ⟹ method won't converge
        return Inf
    elseif D(0.0, β, D_params) < 0.0 || D(0.5maxf, β, D_params) < 0.0 || D(maxf, β, D_params) < 0.0 || T(0.0, α, T_params) < 0.0 || T(0.5finalTime, α, T_params) < 0.0 || T(finalTime, α, T_params) < 0.0 # Check for negative diffusion and delay. More efficient to just check problems at start, midpoint, and endpoint rather than an entire array — should usually be the same.
        return Inf
    else
        try
            SSEArray = get_tmp(SSEArray, first(αβγ))
            MSE = get_tmp(MSE, first(αβγ))
            p = (N, V, Δx, LHS..., RHS..., Du, Ru, D′u, R′u, T, D, R, D′, R′, α, β, γ, D_params, R_params, T_params)
            prob = ODEProblem(sysdegeneral!, convert.(eltype(αβγ), initialCondition), (0.0, finalTime), p)
            SSEArray .= hcat(DifferentialEquations.solve(prob, EQLalg, saveat = δt; PDEkwargs...).u...)
            for j = 1:length(δt)
                for (k, i) in enumerate(iterate_idx[j])
                    val = SSEArray[closest_idx[j][k], j]
                    MSE[i] = abs2((u[i] - val) / σₙ)
                end
            end
            GLSC = obj_scale_GLS(mean(MSE))
            total_loss += GLSC
        catch err
            if err isa InterruptException
                println("Function terminated by user.")
                rethrow(err)
            else
                println(err)
                return Inf
            end
        end
        if isfinite(total_loss)
            errs = get_tmp(errs, first(αβγ))
            DuP = get_tmp(DuP, first(αβγ))
            RuP = get_tmp(RuP, first(αβγ))
            D′uP = get_tmp(D′uP, first(αβγ))
            TuP = get_tmp(TuP, first(αβγ))
            @inbounds for (j, idx) in enumerate(inIdx)
                DuP[idx] = D(f[idx], β, D_params)
                RuP[idx] = R(f[idx], γ, R_params)
                D′uP[idx] = D′(f[idx], β, D_params)
                TuP[idx] = T(unscaled_t̃[idx], α, T_params)
                errs[j] = abs2(fₜ[idx] - TuP[idx] * (D′uP[idx] * fₓ[idx]^2 + DuP[idx] * fₓₓ[idx] + RuP[idx]))
            end
            if DuP[inIdx[1]] < 0.0 || DuP[inIdx[end>>1]] < 0.0 || DuP[inIdx[end]] < 0.0 || TuP[inIdx[1]] < 0.0 || TuP[inIdx[end>>1]] < 0.0 || TuP[inIdx[end]] < 0.0
                return Inf
            else
                PDEC = obj_scale_PDE(mean(errs))
                total_loss += PDEC
            end
        end
    end
    if show_losses
        @printf "Scaled GLS loss contribution: %.6g. Scaled PDE loss contribution: %.6g.\n" GLSC PDEC
    end
    return log(total_loss)
end

"""
    learn_equations!(<arguments>) 

Estimate values for the delay, diffusion, and reaction parameters. See [`bootstrap_gp`](@ref) and [`loss_function`](@ref).

# Arguments
- `x`: The spatial data.
- `t`: The temporal data.
- `u`: The density data.
- `f`: Computed values of `f(x, t)`.
- `fₜ`: Computed values of `fₜ(x, t)`.
- `fₓ`: Computed values of `fₓ(x, t)`.
- `fₓₓ`: Computed values of `fₓₓ(x, t)`.
- `T::Function`: The delay function, given in the form `T(t, α., )`.
- `D::Function`: The diffusion function, given in the form `D(u, β...)`.
- `D′::Function`: The derivative of the diffusion function, given in the form `D′(u, β...)`.
- `R::Function`: The reaction function, given in the form `R(u, γ...)`.
- `R′::Function`: The derivative of the reaction function, given in the form `R′(u, γ, R_params)`.
- `T_params`: Additional known parameters for the delay function.
- `D_params`: Additional known parameters for the diffusion function.
- `R_params`: Additional known parameters for the reaction function. 
- `α`: Initial estimates for the delay parameters.
- `β`: Initial estimates for the diffusion parameters. 
- `γ`: Initial estimates for the reaction parameters.
- `stacked_params`: Matrix which stores parameter values at each optimisation restart. The columns take the form `[α₀; β₀; γ₀]`.
- `lowers`: Lower bounds to use for constructing the Latin hypersquare design, and for the constrained problem if `bootstrap_setup.constrained = true`.
- `uppers`: Upper bounds to use for constructing the Latin hypersquare design, and for the constrained problem if `bootstrap_setup.constrained = true`.
- `constrained`: `true` if the optimisation problems should be constrained, and `false` otherwise.
- `obj_values`: Cache array for storing the objective function values at each optimisation restart.
- `obj_scale_GLS`: The amount by which the GLS loss function should be scaled.
- `obj_scale_PDE`: The amount by which the PDE loss function should be scaled.
- `N`: The number points.
- `V`: The volume of each cell in the spatial mesh.
- `Δx`: The spacing between each point in the spatial mesh.
- `LHS`: Vector defining the left-hand boundary conditions for the PDE. See also the definitions of `(a₀, b₀, c₀)` in [`sysdegeneral!`](@ref).
- `RHS`: Vector defining the right-hand boundary conditions for the PDE. See also the definitions of `(a₁, b₁, c₁)` in [`sysdegeneral!`](@ref).
- `initialCondition`: The initial condition to use for the PDE.
- `finalTime`: The final time to give the solution to the PDE at.
- `alg`: The algorithm to use for solving the discretised PDE.
- `δt`: A vector specifying the times to return the solution to the discretised PDE at.
- `SSEArray`: Cache array for storing the solutions to the PDE. 
- `Du`: Cache array for storing the values of the diffusion function at the mesh points. Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `Ru`: Cache array for storing the values of the reaction function at the mesh points.  Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `D′u`: Cache array used for storing the values of the derivative of the diffusion function `D` at `D′(u)`.
- `R′u`: Cache array used for storing the values of the derivative of the reaction function `R` at `R′(u)`.
- `TuP`: Cache array for storing the values of the delay function at the unscaled times (for the `"PDE"` loss function).  Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `DuP`: Cache array for storing the values of the diffusion function at the estimated density values (for the `"PDE"` loss function).  Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `RuP`: Cache array for storing the values of the reaction function at the estimated density values (for the `"PDE"` loss function).  Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `D′uP`: Cache array for storing the values of the derivative of the diffusion function at the estimated density values (for the `"PDE"` loss function).  Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `RuN`: For storing values of the reaction function at Gauss-Legendre quadrature nodes.  Should be defined as a `PreallocationTools.DiffCache` type; see [`bootstrap_helper`](@ref).
- `inIdx`: Indices in `f` (and `fₜ`) that should be used in the optimisation process. See aso [`data_thresholder`](@ref).
- `unscaled_t̃`: Unscaled `t` values for the bootstrapping grid. 
- `tt`: Number of delay parameters.
- `d`: Number of diffusion parameters.
- `r`: Number of reaction reaction parameters.
- `errs`: Cache array for storing the individual error values. Should be defined as a `PreallocationTools.DiffCache` type.
- `MSE`: Cache array for storing the individual squared errors. Should be defined as a `PreallocationTools.DiffCache` type.
- `optim_setup`: An `Optim.Options` struct used for defining options in `Optim.optimize`.
- `iterate_idx`: Indices to use on the data for finding indices corresponding to specific time values. See the definition in [`bootstrap_gp`](@ref).
- `closest_idx`: Points in the spatial mesh for the ODEs that are closest to the positions in the spatial data `x`. See the definition in [`bootstrap_gp`](@ref).
- `nodes`: The Gauss-Legendre quadrature nodes.
- `weights`: The Gauss-Legendre quadrature weights.
- `show_losses`: `true` if the loss function should be printed to the REPL throughout the optimisation process, and `false` otherwise.
- `σₙ`: The standard deviation of the observation noise, estimated from the Gaussian process.
- `PDEkwargs...`: The keyword arguments to use in `DifferentialEquations.solve`.

# Outputs 
The estimates for the delay parameters, `α`, diffusion parameters, `β`, and reaction parameters, 
`γ`, are updated in-place.
"""
function learn_equations!(x, t, u,
    f, fₜ, fₓ, fₓₓ,
    T, D, D′, R, R′, T_params, D_params, R_params,
    α, β, γ, stacked_params,
    lowers, uppers, constrained, obj_values,
    obj_scale_GLS, obj_scale_PDE,
    N, V, Δx, LHS, RHS, initialCondition, finalTime, alg, δt, SSEArray,
    Du, Ru, D′u, R′u, TuP, DuP, RuP, D′uP, RuN,
    inIdx, unscaled_t̃, tt, d, r,
    errs, MSE, optim_setup,
    iterate_idx, closest_idx, glnodes, glweights, show_losses, σₙ,
    PDEkwargs...)
    #@assert length(obj_values) == size(stacked_params, 2) "The number of objective values must equal the provided number of initial parameter estimate restarts."

    # Define the objective function
    maxf = maximum(f[inIdx])
    optim_fnc = αβγ -> loss_function(αβγ; u,
        f, fₓ, fₓₓ, fₜ,
        N, V, Δx, LHS, RHS,
        T, D, D′, R, R′,
        initialCondition, finalTime, EQLalg = alg,
        δt, SSEArray, Du, Ru, D′u, R′u, TuP, DuP, RuP, D′uP, RuN,
        inIdx, unscaled_t̃, tt, d, r, errs, MSE, obj_scale_GLS, obj_scale_PDE,
        glnodes, glweights, maxf, D_params, R_params, T_params,
        iterate_idx, closest_idx, show_losses, σₙ, PDEkwargs...)

    # Define the optimisation function 
    if constrained
        fit_fnc = αβγ₀ -> Optim.optimize(optim_fnc, lowers, uppers, αβγ₀, Fminbox(LBFGS()), optim_setup; autodiff = :forward)
    else
        fit_fnc = αβγ₀ -> Optim.optimize(optim_fnc, αβγ₀, LBFGS(), optim_setup; autodiff = :forward)
    end

    # Optimise 
    @inbounds for j in 1:size(stacked_params, 2)
        try
            prob = fit_fnc(stacked_params[:, j])
            obj_values[j] = prob.minimum
        catch err
            if err isa InterruptException
                println("Function terminated by user.")
                rethrow(err)
            else
                println(err)
                obj_values[j] = Inf
            end
        end
    end

    min_obj_idx = findmin(obj_values)[2]
    try
        fit_model = fit_fnc(stacked_params[:, min_obj_idx])
        final_params = fit_model.minimizer
        # Allocate parameter values
        α .= @views final_params[1:tt]
        β .= @views final_params[(tt+1):(tt+d)]
        γ .= @views final_params[(tt+d+1):(tt+d+r)]
        return false
    catch
        return true
    end
end