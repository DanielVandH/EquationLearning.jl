# Equation Learning

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://danielvandh.github.io/EquationLearning.jl/dev/home.html) 

This package contains code to perform equation learning with bootstrapping using Gaussian processes, as described in our paper https://www.biorxiv.org/content/10.1101/2022.05.12.491596v1. Currently the method is only implemented for partial differential equations of the form

![equation](http://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20=%20T(t;%20%5Cboldsymbol%7B%5Calpha%7D)%20%5Cleft%5B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(D(u;%20%5Cboldsymbol%7B%5Cbeta%7D)%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%5Cright)%20&plus;%20R(u;%20%5Cboldsymbol%7B%5Cgamma%7D)%5Cright%5D,)

but the ideas could easily be extended to much more complicated problems or other classes of problems. Please see the [documentation](https://danielvandh.github.io/EquationLearning.jl/dev/home.html) for more details and instructions for use. In this documentation we also provide details for reproducing the results in our paper. It is planned to later extend this package to include more general methods for model discovery than just those described in the paper above.

# Installation 

To install the package, you can use:
```
] add https://github.com/DanielVandH/EquationLearning.jl.git
```
in the Julia REPL. Note that the `]` prefix is to enter the Pkg REPL.

# Issues 

Any suggestions, questions, and/or issues with the package should be given as an issue, or as an email to [Daniel VandenHeuvel](mailto:vandenh2@qut.edu.au?subject=GP%20Equation%20Learning&body=Dear%20Daniel,) (this link opens an email to vandenh2@qut.edu.au). Issues are preferred.

At the time of writing, the Issues tab is essentially being used by myself as a to-do list. Feel free to comment or ask about these features as well.
