# Equation Learning 

[Click here to go back to the repository](https://github.com/DanielVandH/EquationLearning.jl).

```@contents
```

This package allows for the learning of mechanisms $T(t; \mathbf{\alpha})$, $D(u; \mathbf{\beta})$, and $R(u; \mathbf{\gamma})$ for 
delay, diffusion, and reaction model, for some cell data $u$ and points $(x, t)$, assuming that $u$ satisfies the model

```math
\frac{\partial u}{\partial t} = T(t; \boldsymbol{\alpha})\left[\frac{\partial}{\partial x}\left(D(u; \boldsymbol{\beta})\frac{\partial u}{\partial x}\right) + R(u; \boldsymbol{\gamma})\right].
```

The package fits a Gaussian process to the data $u$ at these points $(x, t)$ and uses it to draw samples from, allowing for multiple estimates of the parameters $\mathbf{\alpha}$, $\mathbf{\beta}$, and $\mathbf{\gamma}$ to be obtained, thus providing uncertainty quantification for these learned mechanisms. See our paper ... for more details. The main function exported by this package is `bootstrap_gp` which actual fits a given model with uncertainty quantification.
