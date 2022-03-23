# Equation Learning

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://DanielVandH.github.io/EquationLearning.jl/dev)

This package contains code to perform equation learning with bootstrapping using Gaussian processes, as described in our paper <...>. Currently the method is only implemented for partial differential equations of the form

![equation](http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20=%20T(t;%20%5Cmathbf%7B%5Calpha%7D)%20%5Cleft%5B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(D(u;%20%5Cmathbf%7B%5Cbeta%7D)%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%5Cright)%20&plus;%20R(u;%20%5Cmathbf%7B%5Cgamma%7D)%5Cright%5D,)

but the ideas could easily be extended to much more complicated problems or other classes of problems. Please see the [documentation](https://DanielVandH.github.io/EquationLearning.jl/dev) for more details and instructions for use. 

# Installation 

To install the package, you can use:
```
] add https://github.com/DanielVandH/EquationLearning.jl.git
```
in the Julia REPL. Note that the `]` prefix is to enter the Pkg REPL.

# Issues 

Any questions or issues with the package should be given as an issue, or as an email to [Daniel VandenHeuvel](mailto:vandenh2@qut.edu.au?subject=GP%20Equation%20Learning&body=Dear%20Daniel,) (this link opens an email to vandenh2@qut.edu.au). Issues are preferred.

# The Future 

Some future plans (amongst many other ideas):

- Top priority: Implement reliable unit tests.
- Devise better methods for terminating the optimiser faster when the initial parameter estimate is clearly not close to optimal, e.g. by limiting the allowed time in the optimiser and implementing a callback in the ODE solver that will more effectively detect when the solution needs to terminate. This would significantly improve the runtime of the algorithm.
- Give better default options for the ODE solvers. Would be good to have a method comparable in speed to Sundials' `CVODE_BDF(linear_solver = :Band, jac_upper = 1, jac_lower = 1)` but permits automatic differentiation for the optimiser.
- Make the structure of the arguments more user-friendly rather than using a bunch of structures.
- Parallelise the code and change the bootstrapping loop to a `for` loop rather than a `while` loop. 
- Implement the biologically-informed neural networks of [Lagergren et al. (2020)](https://doi.org/10.1371/journal.pcbi.1008462).
- Implement the Bayes-PDE Find method of [Martina-Perez et al. (2021)](https://doi.org/10.1098/rspa.2021.0426).
