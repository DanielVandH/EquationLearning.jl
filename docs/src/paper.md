# VandenHeuvel et al. (2022)

This section briefly discusses our paper, and steps for reproducing the figures in the paper. The paper 
can be found here ..., and the abstract of the paper is:

> Parameter estimation for mathematical models of biological processes is often difficult and depends significantly on the quality and quantity of available data. We introduce an efficient framework using Gaussian processes to discover mechanisms underlying delay, migration, and proliferation in a cell invasion experiment. Gaussian processes are leveraged with bootstrapping to provide uncertainty quantification for the mechanisms that drive the invasion process. Our framework is efficient, parallelisable, and can be applied to other biological problems. We illustrate our methods using a canonical scratch assay experiment, demonstrating how simply we can explore different functional forms and develop and test hypotheses about underlying mechanisms, such as whether delay is present. All code and data to reproduce this work is available at https://github.com/DanielVandH/EquationLearning.jl.

The scratch assay data from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040) can be found in this GitHub repository in [VandenHeuvel2022_PaperCode/data](https://github.com/DanielVandH/EquationLearning.jl/blob/main/VandenHeuvel2022_PaperCode/data).

## Paper results

The main body of the paper is produced using the code in [VandenHeuvel2022_PaperCode/paper_code.jl](https://github.com/DanielVandH/EquationLearning.jl/blob/5466b87ae7ed3d3d171123ddf3d595d881538490/VandenHeuvel2022_PaperCode/paper_code.jl). Below, we list the sections that this script is broken into, along with descriptions of these sections:

1. *Load the required packages*

    Here we simply load all the necessary packages.

2. *Set some global parameters* 

    (Note that we also have a section called this in 7. below.) This section defines some parameters for plotting that are used in most of the sections. We also write:

    ```julia
    LinearAlgebra.BLAS.set_num_threads(1)
    ```

    This setting was used to remove issues relating to `A \ b` giving `StackOverflowError`. See, for example, [#43301](https://github.com/JuliaLang/julia/issues/43301) or [#43242](https://github.com/JuliaLang/julia/issues/43242).

3. *Read in the data from Jin et al. (2016)*

    Here we read in the data from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040), scaling the data by $\hat x = 1000$ and $\hat t = 24$.

4. *Figure X: Plotting the density data from Jin et al. (2016)*

    This code plots the data from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040) and also plots a curve through the average of the experimental replicates at each point in sspace and time.

5. *Figure X: Plotting the Gaussian processes fit to the data from Jin et al. (2016)*

    This section plots the Gaussian processes over the data from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040), These Gaussian processes are fit using [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl).

6. *Figure X: Plotting the space-time diagram for the Gaussian process*

    This section plots the same Gaussian processes, but now plots them on the $(x, t)$ place, colouring the points by the mean of the Gaussian process posterior at each point $(x, t)$.

7. *Set some global parameters* 

    This section now defines the global parameters for the bootstrapping. The parameters for the PDE are defined first, and then the parameters for bootstrapping. We also remove the left-most points from the data from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040) here.

8. *Model fits*

    This section contains the actual code that gives the figures in the paper. There are five functions that we define first:

    - `model_fits`: This function fits, for a given dataset, a Fisher-Kolmogorov model (with and without delay), a Porous-Fisher model (with and without delay), and a delayed generalised Porous-FKPP model.
    - `plot_fisher_kolmogorov_delay`: This function plots the results from a delayed Fisher-Kolmogorov model.
    - `plot_generalised_fkpp_delay`: This function plots the results from a delayed generalised Porous-FKPP model.
    - `plot_pde_soln!`: This function adds, to an existing figure, an axis for the PDE solutions for a given dataset.
    - `plot_pde_soln`: For the six datasets, this function plots all of the PDE solutions from each dataset on the same figure.

    After these functions are defined, we define parameters that scale each parameter for each function such that the scaled parameters that we have to estimate are all $\mathcal O(1)$. We based these parameter scales on the results from [Jin et al. (2016)](https://doi.org/10.1016/j.jtbi.2015.10.040) and [Lagergren et al. (2020)](https://doi.org/10.1371/journal.pcbi.1008462), or adjusted further based on issues we observed when fitting models. We then fit all the models, which takes a reasonably long time to complete. We then make all the plots.

## Simulation studies

We also present several simulation studies in the paper, all of which are given in [VandenHeuvel2022_PaperCode/simulation_studies.jl](https://github.com/DanielVandH/EquationLearning.jl/blob/5466b87ae7ed3d3d171123ddf3d595d881538490/VandenHeuvel2022_PaperCode/simulation_studies.jl).

### Simulation study I: Fisher-Kolmogorov Model, 10,000 cells per well

In this study we fit some models to data simulated from the Fisher-Kolmogorov model

```math 
\frac{\partial u}{\partial t} = \beta_1\frac{\partial^2u}{\partial x^2} + \gamma_1u\left(1-\frac{u}{K}\right).
```

Running all this code will produce the figures in the corresponding section of our paper.

### Simulation study I: Fisher-Kolmogorov Model, 10,000 cells per well

In this study we fit some models to data simulated from the Fisher-Kolmogorov model

```math 
\frac{\partial u}{\partial t} = \beta_1\frac{\partial^2u}{\partial x^2} + \gamma_1u\left(1-\frac{u}{K}\right).
```

Running all this code will produce the figures in the corresponding section of our paper.

### Simulation study II: Fisher-Kolmogorov Model with delay, 10,000 cells per well

In this study we fit some models to data simulated from the delayed Fisher-Kolmogorov model

```math 
\frac{\partial u}{\partial t} = \frac{1}{1+\exp(-\alpha_1-\alpha_2t)}\left[\beta_1\frac{\partial^2u}{\partial x^2} + \gamma_1u\left(1-\frac{u}{K}\right)\right].
```

Running all this code will produce the figures in the corresponding section of our paper.

### Simulation study III: Fisher Kolmogorov model, 10,000 cells per well, basis function approach 

This study fits the same model as in study I, but using the basis function approach. Running all this code will produce the figures in the corresponding section of our paper.

### Simulation study IV: Data thresholding on the Fisher-Kolmogorov model of Study I 

This study considers the effects of data thresholding on the model in the first study. This study is done by simply looping over many tolerance values. Running all this code will produce the figures in the corresponding section of our paper.

### Simulation study V: Data thresholding on the Fisher-Kolmogorov model of Study II

This study considers the effects of data thresholding on the model in the second study. This study is done by simply looping over many tolerance values. Running all this code will produce the figures in the corresponding section of our paper.



