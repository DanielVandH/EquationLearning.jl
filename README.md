# Equation Learning

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://danielvandh.github.io/EquationLearning.jl/dev/home.html) [![codecov](https://codecov.io/gh/DanielVandH/EquationLearning.jl/branch/main/graph/badge.svg?token=0C6HHS1Q9F)](https://codecov.io/gh/DanielVandH/EquationLearning.jl)

This package contains code to perform equation learning with bootstrapping using Gaussian processes, as described in our paper <...>. Currently the method is only implemented for partial differential equations of the form

![equation](http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20=%20T(t;%20%5Cboldsymbol%7B%5Calpha%7D)%20%5Cleft%5B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(D(u;%20%5Cboldsymbol%7B%5Cbeta%7D)%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%5Cright)%20&plus;%20R(u;%20%5Cboldsymbol%7B%5Cgamma%7D)%5Cright%5D,)

but the ideas could easily be extended to much more complicated problems or other classes of problems. Please see the [documentation](https://danielvandh.github.io/EquationLearning.jl/dev/home.html) for more details and instructions for use. In this documentation we also provide details for reproducing the results in our paper.

# Installation 

To install the package, you can use:
```
] add https://github.com/DanielVandH/EquationLearning.jl.git
```
in the Julia REPL. Note that the `]` prefix is to enter the Pkg REPL.

# Issues 

Any questions or issues with the package should be given as an issue, or as an email to [Daniel VandenHeuvel](mailto:vandenh2@qut.edu.au?subject=GP%20Equation%20Learning&body=Dear%20Daniel,) (this link opens an email to vandenh2@qut.edu.au). Issues are preferred.
