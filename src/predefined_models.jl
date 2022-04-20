#####################################################################
## Script description: predefined_models.jl
##
## This script contains certain functions used for fitting bootstrapped models 
## for some predefined model types. 
##
#####################################################################
function fisher_kolmogorov(x, t, u, K, lowers, uppers;
    delay=false, basis=false,
    gp_setup::GP_Setup=GP_Setup(u),
    bootstrap_setup::Bootstrap_Setup=Bootstrap_Setup(x, t, u),
    optim_setup::Optim.Options=Optim.Options(),
    pde_setup::PDE_Setup=PDE_Setup(x),
    D_params=[1.0], R_params=[K, 1.0], T_params=delay ? [1.0, 1.0] : nothing, zvals=nothing,
    PDEkwargs...)
    if basis
        D = convert(Vector{Function}, [(u, p) -> 1.0])
        D′ = convert(Vector{Function}, [(u, p) -> 0.0])
        R = convert(Vector{Function}, [(u, p) -> u * (1.0 - u / p[2])])
        R′ = convert(Vector{Function}, [(u, p) -> 1.0 - 2u / p[2]])
    end
end

function porous_fisher(x, t, u, K;
    gp_setup::GP_Setup=GP_Setup(u),
    bootstrap_setup::Bootstrap_Setup=Bootstrap_Setup(x, t, u),
    optim_setup::Optim.Options=Optim.Options(),
    pde_setup::PDE_Setup=PDE_Setup(x),
    D_params=nothing, R_params=nothing, T_params=nothing, zvals=nothing,
    delay=false, basis=false, PDEkwargs...)

end

function generalised_porous_fkpp(x, t, u, K;
    gp_setup::GP_Setup=GP_Setup(u),
    bootstrap_setup::Bootstrap_Setup=Bootstrap_Setup(x, t, u),
    optim_setup::Optim.Options=Optim.Options(),
    pde_setup::PDE_Setup=PDE_Setup(x),
    D_params=nothing, R_params=nothing, T_params=nothing, zvals=nothing,
    delay=false, PDEkwargs...)

end