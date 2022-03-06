# EquationLearning

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://DanielVandH.github.io/EquationLearning.jl/dev)

This package contains code to perform equation learning with bootstrapping using Gaussian processes, as described in our paper <...>. Currently the method is only implemented for partial differential equations of the form

![equation](http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20=%20T(t;%20%5Cmathbf%7B%5Calpha%7D)%20%5Cleft%5B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(D(u;%20%5Cmathbf%7B%5Cbeta%7D)%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%5Cright)%20&plus;%20R(u;%20%5Cmathbf%7B%5Cgamma%7D)%5Cright%5D,)

but the ideas could easily be extended to much more complicated problems or other classes of problems.

To install the package, you can use:
```
] add https://github.com/DanielVandH/EquationLearning.jl.git
```
in the Julia REPL. Note that the `]` prefix is to enter the Pkg REPL.
